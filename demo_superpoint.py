#!/usr/bin/env python
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2018
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Daniel DeTone (ddetone)
#                       Tomasz Malisiewicz (tmalisiewicz)
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%


import argparse
import csv
import glob
import numpy as np
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import torch
import yaml

from models import SUPERPOINT_MODEL_CHOICES, build_superpoint_model

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

# Jet colormap for visualization.
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

class SuperPointFrontend(object):
  """ Wrapper around pytorch net to help with pre and post image processing. """
  def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh,
               cuda=False, mps=False, model_name='MagicLeap'):
    self.name = 'SuperPoint'
    if cuda and mps:
      raise ValueError('Cannot enable both CUDA and MPS backends.')
    self.cuda = cuda
    self.mps = mps
    if self.mps:
      mps_backend = getattr(torch.backends, 'mps', None)
      if mps_backend is None or not mps_backend.is_available():
        raise ValueError('MPS backend requested but not available in this PyTorch build.')
    if self.cuda:
      self.device = torch.device('cuda')
    elif self.mps:
      self.device = torch.device('mps')
    else:
      self.device = torch.device('cpu')
    print('Running {} on device: {}'.format(self.name, self.device))
    self.nms_dist = nms_dist
    self.conf_thresh = conf_thresh
    self.nn_thresh = nn_thresh # L2 descriptor distance for good match.
    self.cell = 8 # Size of each output cell. Keep this fixed.
    self.border_remove = 4 # Remove points this close to the border.

    # Load the network in inference mode.
    self.net = build_superpoint_model(model_name, weights_path, self.device)

  def nms_fast(self, in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
  
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
  
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
  
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
  
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
      grid[rcorners[1,i], rcorners[0,i]] = 1
      inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

  def run(self, img):
    """ Process a numpy image to extract points and descriptors.
    Input
      img - HxW numpy float32 input image in range [0,1].
    Output
      corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      desc - 256xN numpy array of corresponding unit normalized descriptors.
      heatmap - HxW numpy heatmap in range [0,1] of point confidences.
      """
    assert img.ndim == 2, 'Image must be grayscale.'
    assert img.dtype == np.float32, 'Image must be float32.'
    H, W = img.shape[0], img.shape[1]
    inp = img.copy()
    inp = (inp.reshape(1, H, W))
    inp = torch.from_numpy(inp)
    inp = torch.autograd.Variable(inp).view(1, 1, H, W)
    if self.device.type != 'cpu':
      inp = inp.to(self.device)
    # Forward pass of network.
    outs = self.net.forward(inp)
    semi, coarse_desc = outs[0], outs[1]
    # Convert pytorch -> numpy.
    semi = semi.data.cpu().numpy().squeeze()
    # --- Process points.
    dense = np.exp(semi) # Softmax.
    dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
    # Remove dustbin.
    nodust = dense[:-1, :, :]
    # Reshape to get full resolution heatmap.
    Hc = int(H / self.cell)
    Wc = int(W / self.cell)
    nodust = nodust.transpose(1, 2, 0)
    heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
    heatmap = np.reshape(heatmap, [Hc*self.cell, Wc*self.cell])
    xs, ys = np.where(heatmap >= self.conf_thresh) # Confidence threshold.
    if len(xs) == 0:
      return np.zeros((3, 0)), None, None
    pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist) # Apply NMS.
    inds = np.argsort(pts[2,:])
    pts = pts[:,inds[::-1]] # Sort by confidence.
    # Remove points along border.
    bord = self.border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    # --- Process descriptor.
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
      desc = np.zeros((D, 0))
    else:
      # Interpolate into descriptor map using 2D point locations.
      samp_pts = torch.from_numpy(pts[:2, :].copy())
      samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
      samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
      samp_pts = samp_pts.transpose(0, 1).contiguous()
      samp_pts = samp_pts.view(1, 1, -1, 2)
      samp_pts = samp_pts.float()
      if self.device.type != 'cpu':
        samp_pts = samp_pts.to(self.device)
      desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
      desc = desc.data.cpu().numpy().reshape(D, -1)
      desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return pts, desc, heatmap


class PointTracker(object):
  """ Class to manage a fixed memory of points and descriptors that enables
  sparse optical flow point tracking.

  Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
  tracks with maximum length L, where each row corresponds to:
  row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
  """

  def __init__(self, max_length, nn_thresh):
    if max_length < 2:
      raise ValueError('max_length must be greater than or equal to 2.')
    self.maxl = max_length
    self.nn_thresh = nn_thresh
    self.all_pts = []
    for _ in range(self.maxl):
      self.all_pts.append(np.zeros((2, 0)))
    self.last_desc = None
    self.tracks = np.zeros((0, self.maxl+2))
    self.track_count = 0
    self.max_score = 9999

  def save_state(self) -> Dict[str, Any]:
    """Create a deep copy snapshot of the tracker buffers."""
    return {
        'last_desc': None if self.last_desc is None else self.last_desc.copy(),
        'all_pts': [pts.copy() for pts in self.all_pts],
        'tracks': self.tracks.copy(),
        'track_count': self.track_count,
        'nn_thresh': self.nn_thresh,
    }

  def load_state(self, state: Dict[str, Any]) -> None:
    """Restore the tracker buffers from a snapshot created with save_state."""
    last_desc = state.get('last_desc')
    self.last_desc = None if last_desc is None else last_desc.copy()
    self.all_pts = [pts.copy() for pts in state['all_pts']]
    self.tracks = state['tracks'].copy()
    self.track_count = state['track_count']
    self.nn_thresh = state['nn_thresh']

  def nn_match_two_way(self, desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
      return np.zeros((3, 0))
    if nn_thresh < 0.0:
      raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches

  def get_offsets(self):
    """ Iterate through list of points and accumulate an offset value. Used to
    index the global point IDs into the list of points.

    Returns
      offsets - N length array with integer offset locations.
    """
    # Compute id offsets.
    offsets = []
    offsets.append(0)
    for i in range(len(self.all_pts)-1): # Skip last camera size, not needed.
      offsets.append(self.all_pts[i].shape[1])
    offsets = np.array(offsets)
    offsets = np.cumsum(offsets)
    return offsets

  def update(self, pts, desc):
    """ Add a new set of point and descriptor observations to the tracker.

    Inputs
      pts - 3xN numpy array of 2D point observations.
      desc - DxN numpy array of corresponding D dimensional descriptors.
    """
    if pts is None or desc is None:
      print('PointTracker: Warning, no points were added to tracker.')
      return
    assert pts.shape[1] == desc.shape[1]
    # Initialize last_desc.
    if self.last_desc is None:
      self.last_desc = np.zeros((desc.shape[0], 0))
    # Remove oldest points, store its size to update ids later.
    remove_size = self.all_pts[0].shape[1]
    self.all_pts.pop(0)
    self.all_pts.append(pts)
    # Remove oldest point in track.
    self.tracks = np.delete(self.tracks, 2, axis=1)
    # Update track offsets.
    for i in range(2, self.tracks.shape[1]):
      self.tracks[:, i] -= remove_size
    self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
    offsets = self.get_offsets()
    # Add a new -1 column.
    self.tracks = np.hstack((self.tracks, -1*np.ones((self.tracks.shape[0], 1))))
    # Try to append to existing tracks.
    matched = np.zeros((pts.shape[1])).astype(bool)
    matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
    for match in matches.T:
      # Add a new point to it's matched track.
      id1 = int(match[0]) + offsets[-2]
      id2 = int(match[1]) + offsets[-1]
      found = np.argwhere(self.tracks[:, -2] == id1)
      if found.shape[0] > 0:
        matched[int(match[1])] = True
        row = int(found)
        self.tracks[row, -1] = id2
        if self.tracks[row, 1] == self.max_score:
          # Initialize track score.
          self.tracks[row, 1] = match[2]
        else:
          # Update track score with running average.
          # NOTE(dd): this running average can contain scores from old matches
          #           not contained in last max_length track points.
          track_len = (self.tracks[row, 2:] != -1).sum() - 1.
          frac = 1. / float(track_len)
          self.tracks[row, 1] = (1.-frac)*self.tracks[row, 1] + frac*match[2]
    # Add unmatched tracks.
    new_ids = np.arange(pts.shape[1]) + offsets[-1]
    new_ids = new_ids[~matched]
    new_tracks = -1*np.ones((new_ids.shape[0], self.maxl + 2))
    new_tracks[:, -1] = new_ids
    new_num = new_ids.shape[0]
    new_trackids = self.track_count + np.arange(new_num)
    new_tracks[:, 0] = new_trackids
    new_tracks[:, 1] = self.max_score*np.ones(new_ids.shape[0])
    self.tracks = np.vstack((self.tracks, new_tracks))
    self.track_count += new_num # Update the track count.
    # Remove empty tracks.
    keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
    self.tracks = self.tracks[keep_rows, :]
    # Store the last descriptors.
    self.last_desc = desc.copy()
    return

  def get_tracks(self, min_length):
    """ Retrieve point tracks of a given minimum length.
    Input
      min_length - integer >= 1 with minimum track length
    Output
      returned_tracks - M x (2+L) sized matrix storing track indices, where
        M is the number of tracks and L is the maximum track length.
    """
    if min_length < 1:
      raise ValueError('\'min_length\' too small.')
    valid = np.ones((self.tracks.shape[0])).astype(bool)
    good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
    # Remove tracks which do not have an observation in most recent frame.
    not_headless = (self.tracks[:, -1] != -1)
    keepers = np.logical_and.reduce((valid, good_len, not_headless))
    returned_tracks = self.tracks[keepers, :].copy()
    return returned_tracks

  def draw_tracks(self, out, tracks):
    """ Visualize tracks all overlayed on a single image.
    Inputs
      out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
      tracks - M x (2+L) sized matrix storing track info.
    """
    # Store the number of points per camera.
    pts_mem = self.all_pts
    N = len(pts_mem) # Number of cameras/images.
    # Get offset ids needed to reference into pts_mem.
    offsets = self.get_offsets()
    # Width of track and point circles to be drawn.
    stroke = 1
    # Iterate through each track and draw it.
    for track in tracks:
      clr = myjet[int(np.clip(np.floor(track[1]*10), 0, 9)), :]*255
      for i in range(N-1):
        if track[i+2] == -1 or track[i+3] == -1:
          continue
        offset1 = offsets[i]
        offset2 = offsets[i+1]
        idx1 = int(track[i+2]-offset1)
        idx2 = int(track[i+3]-offset2)
        pt1 = pts_mem[i][:2, idx1]
        pt2 = pts_mem[i+1][:2, idx2]
        p1 = (int(round(pt1[0])), int(round(pt1[1])))
        p2 = (int(round(pt2[0])), int(round(pt2[1])))
        cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
        # Draw end points of each track.
        if i == N-2:
          clr2 = (255, 0, 0)
          cv2.circle(out, p2, stroke, clr2, -1, lineType=16)


def get_untracked_points_mask(pts: np.ndarray, tracks: np.ndarray, tracker: 'PointTracker') -> np.ndarray:
  """Calculate mask for points that are not currently tracked.

  Args:
    pts: Array of current frame points shaped 3xN.
    tracks: Matrix with active track metadata.
    tracker: Tracker managing global point identifiers.

  Returns:
    Boolean mask where True marks points that are not yet part of a track.
  """
  num_pts = pts.shape[1]
  if num_pts == 0:
    return np.ones((0,), dtype=bool)

  if tracks.size == 0:
    return np.ones(num_pts, dtype=bool)

  offsets = tracker.get_offsets()
  if offsets.size == 0:
    return np.ones(num_pts, dtype=bool)

  untracked_mask = np.ones(num_pts, dtype=bool)
  current_offset = offsets[-1]
  current_tracked_ids = tracks[:, -1].astype(np.int64)
  current_tracked_ids = current_tracked_ids[current_tracked_ids >= 0]
  if current_tracked_ids.size == 0:
    return untracked_mask

  local_indices = current_tracked_ids - current_offset
  valid_local = local_indices[(local_indices >= 0) & (local_indices < num_pts)]
  if valid_local.size == 0:
    return untracked_mask

  untracked_mask[valid_local] = False
  return untracked_mask


class VideoStreamer(object):
  """ Class to help process image streams. Three types of possible inputs:"
    1.) USB Webcam.
    2.) A directory of images (files in directory matching 'img_glob').
    3.) A video file, such as an .mp4 or .avi file.
  """
  def __init__(self, basedir, camid, height, width, skip, img_glob):
    self.cap = []
    self.camera = False
    self.video_file = False
    self.listing = []
    self.sizer = [height, width]
    self.i = 0
    self.skip = skip
    self.maxlen = 1000000
    self.total_frames = None
    self.last_name = ''
    # If the "basedir" string is the word camera, then use a webcam.
    if basedir == "camera/" or basedir == "camera":
      print('==> Processing Webcam Input.')
      self.cap = cv2.VideoCapture(camid)
      self.listing = range(0, self.maxlen)
      self.camera = True
    else:
      # Try to open as a video first.
      input_path = Path(basedir)
      self.cap = cv2.VideoCapture(basedir)
      capture_opened = hasattr(self.cap, 'isOpened') and self.cap.isOpened()
      is_video_file = capture_opened and (input_path.is_file() or input_path.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'))

      if is_video_file:
        print('==> Processing Video Input.')
        num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames <= 0:
          num_frames = self.maxlen
          self.total_frames = None
        else:
          self.total_frames = num_frames
        self.listing = range(0, num_frames)
        self.listing = self.listing[::self.skip]
        self.camera = True
        self.video_file = True
        self.maxlen = len(self.listing) if num_frames != self.maxlen else self.maxlen
      else:
        if capture_opened:
          self.cap.release()
        print('==> Processing Image Directory Input.')
        search = os.path.join(basedir, img_glob)
        self.listing = glob.glob(search)
        self.listing.sort()
        self.listing = self.listing[::self.skip]
        self.maxlen = len(self.listing)
        if self.maxlen == 0:
          raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')
        self.total_frames = self.maxlen

  def read_image(self, impath, img_size):
    """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    grayim = cv2.imread(impath, 0)
    if grayim is None:
      raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    interp = cv2.INTER_AREA
    grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    return grayim

  def next_frame(self):
    """ Return the next frame, and increment internal counter.
    Returns
       image: Next H x W image.
       status: True or False depending whether image was loaded.
    """
    if self.i == self.maxlen:
      return (None, False)
    if self.camera:
      ret, input_image = self.cap.read()
      if ret is False:
        print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
        return (None, False)
      if self.video_file:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
        name = 'frame_%06d' % self.listing[self.i]
      else:
        name = 'camera_%06d' % self.i
      input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                               interpolation=cv2.INTER_AREA)
      input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
      input_image = input_image.astype('float')/255.0
    else:
      image_file = self.listing[self.i]
      input_image = self.read_image(image_file, self.sizer)
      name = os.path.basename(image_file)
    # Increment internal counter.
    self.i = self.i + 1
    self.last_name = name
    input_image = input_image.astype('float32')
    return (input_image, True)


if __name__ == '__main__':

  # Parse command line arguments.
  config_parser = argparse.ArgumentParser(add_help=False)
  config_parser.add_argument('--config', type=str,
      help='YAML file with default options (keys must match CLI flags).')

  parser = argparse.ArgumentParser(parents=[config_parser],
                                   description='PyTorch SuperPoint Demo.')
  parser.add_argument('input', type=str, nargs='?', default='',
      help='Image directory or movie file or "camera" (for webcam).')
  parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
      help='Path to pretrained weights file (default: superpoint_v1.pth).')
  parser.add_argument('--model', type=str, default='MagicLeap',
      choices=SUPERPOINT_MODEL_CHOICES,
      help='SuperPoint model variant to load (default: MagicLeap).')
  parser.add_argument('--img_glob', type=str, default='*.png',
      help='Glob match if directory of images is specified (default: \'*.png\').')
  parser.add_argument('--skip', type=int, default=1,
      help='Images to skip if input is movie or directory (default: 1).')
  parser.add_argument('--show_extra', action='store_true',
      help='Show extra debug outputs (default: False).')
  parser.add_argument('--H', type=int, default=120,
      help='Input image height (default: 120).')
  parser.add_argument('--W', type=int, default=160,
      help='Input image width (default:160).')
  parser.add_argument('--display_scale', type=int, default=2,
      help='Factor to scale output visualization (default: 2).')
  parser.add_argument('--min_length', type=int, default=2,
      help='Minimum length of point tracks (default: 2).')
  parser.add_argument('--max_length', type=int, default=90,
      help='Maximum length of point tracks (default: 90).')
  parser.add_argument('--show_keypoints', action='store_true',
      help='Show detected keypoints that are not currently tracked (default: False).')
  parser.add_argument('--nms_dist', type=int, default=4,
      help='Non Maximum Suppression (NMS) distance (default: 4).')
  parser.add_argument('--conf_thresh', type=float, default=0.015,
      help='Detector confidence threshold (default: 0.015).')
  parser.add_argument('--nn_thresh', type=float, default=0.7,
      help='Descriptor matching threshold (default: 0.7).')
  parser.add_argument('--camid', type=int, default=0,
      help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
  parser.add_argument('--waitkey', type=int, default=1,
      help='OpenCV waitkey time in ms (default: 1).')
  parser.add_argument('--cuda', action='store_true',
      help='Use cuda GPU to speed up network processing speed (default: False)')
  parser.add_argument('--mps', action='store_true',
      help='Use Apple Metal Performance Shaders backend (default: False).')
  parser.add_argument('--no_display', action='store_true',
      help='Do not display images to screen. Useful if running remotely (default: False).')
  parser.add_argument('--report', action='store_true',
      help='When used with --no_display, generate a metrics report (default: False).')
  parser.add_argument('--report_name', type=str, default='report',
      help='Name prefix for the report folder under ./reports (default: report).')
  parser.add_argument('--write', action='store_true',
      help='Save output frames to a directory (default: False)')
  parser.add_argument('--write_dir', type=str, default='tracker_outputs/',
      help='Directory where to write output frames (default: tracker_outputs/).')

  config_args, remaining = config_parser.parse_known_args()
  if config_args.config:
    config_path = Path(config_args.config)
    if not config_path.exists():
      parser.error('Config file not found: {}'.format(config_path))
    config_data = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(config_data, dict):
      parser.error('Config file must map option names to values.')
    parser.set_defaults(**config_data)

  opt = parser.parse_args(remaining)
  opt.config = config_args.config
  if opt.cuda and opt.mps:
    parser.error('Choose only one accelerator: either --cuda or --mps.')
  if opt.mps:
    mps_backend = getattr(torch.backends, 'mps', None)
    if mps_backend is None or not mps_backend.is_available():
      parser.error('MPS backend requested but not available in this PyTorch build.')
  if opt.report and not opt.no_display:
    parser.error('--report requires --no_display')
  if not opt.input:
    parser.error('No input specified. Provide a source on the command line or in the YAML config via "input".')
  print(opt)

  show_keypoints = opt.show_keypoints or opt.write

  # This class helps load input images from different sources.
  vs = VideoStreamer(opt.input, opt.camid, opt.H, opt.W, opt.skip, opt.img_glob)

  print('==> Loading pre-trained network.')
  # This class runs the SuperPoint network and processes its outputs.
  fe = SuperPointFrontend(weights_path=opt.weights_path,
                          nms_dist=opt.nms_dist,
                          conf_thresh=opt.conf_thresh,
                          nn_thresh=opt.nn_thresh,
                          cuda=opt.cuda,
                          mps=opt.mps,
                          model_name=opt.model)
  print('==> Successfully loaded pre-trained network.')

  # This class helps merge consecutive point matches into tracks.
  tracker = PointTracker(opt.max_length, nn_thresh=fe.nn_thresh)

  # Create a window to display the demo.
  info_win = None
  if not opt.no_display:
    win_label = Path(opt.weights_path).name if opt.weights_path else 'SuperPoint Tracker'
    help_label = f'{win_label} - Help' if opt.weights_path else 'SuperPoint Help'
    win = win_label
    info_win = help_label
    cv2.namedWindow(win)
    cv2.namedWindow(info_win, cv2.WINDOW_AUTOSIZE)
  else:
    print('Skipping visualization, will not show a GUI.')

  # Font parameters for visualizaton.
  font = cv2.FONT_HERSHEY_DUPLEX
  font_clr = (255, 255, 255)
  font_pt = (4, 12)
  font_sc = 0.4

  # Create output directory if desired.
  if opt.write:
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    base_write_dir = Path(opt.write_dir)
    base_write_dir.mkdir(parents=True, exist_ok=True)
    run_write_dir = base_write_dir / timestamp
    suffix = 1
    while run_write_dir.exists():
      run_write_dir = base_write_dir / f'{timestamp}_{suffix:02d}'
      suffix += 1
    run_write_dir.mkdir()
    print('==> Will write outputs to %s' % run_write_dir)
    opt.write_dir = str(run_write_dir)

    if opt.config:
      config_path = Path(opt.config)
      if config_path.exists():
        config_copy_path = run_write_dir / f'config_{timestamp}.yaml'
        config_copy_path.write_text(config_path.read_text(encoding='utf-8'),
                                   encoding='utf-8')
        print('==> Saved config snapshot to %s' % config_copy_path)
      else:
        print('==> Config file not found, skipping copy: {}'.format(config_path))

  report_enabled = opt.no_display and opt.report
  report_rows: List[Dict[str, Any]] = []
  keypoint_counts: List[int] = []
  keypoint_conf_all: List[float] = []
  tracked_confidences_all: List[float] = []
  tracked_scores_for_corr: List[float] = []
  track_counts: List[int] = []
  track_lengths_all: List[int] = []
  track_scores_all: List[float] = []
  untracked_ratios: List[float] = []
  forward_times_ms: List[float] = []
  total_times_ms: List[float] = []
  track_stats: Dict[int, Dict[str, float]] = {}

  timestamp = time.strftime('%Y%m%d-%H%M%S')
  report_name = None
  if report_enabled:
    base_report_dir = Path('reports')
    base_report_dir.mkdir(parents=True, exist_ok=True)
    label = (opt.report_name or 'report').strip()
    if not label:
      label = 'report'
    label = label.replace(' ', '_')
    label = Path(label).name
    report_dir = base_report_dir / f'{label}_{timestamp}'
    suffix = 1
    while report_dir.exists():
      report_dir = base_report_dir / f'{label}_{timestamp}_{suffix:02d}'
      suffix += 1
    report_dir.mkdir(parents=True, exist_ok=True)

  report_total_frames = vs.total_frames if report_enabled else None
  progress_bar_width = 30
  progress_state = {'last_fraction': -1.0}

  def build_visualization(frame_state: Dict[str, Any], draw_keypoints: bool) -> np.ndarray:
    """Create visualization mosaics using cached frame data.

    Args:
      frame_state: Cached tensors and arrays describing the last processed frame.
      draw_keypoints: Flag deciding whether to render untracked keypoints.

    Returns:
      Visualization image ready for display or export.
    """
    img = frame_state['img']
    pts = frame_state['pts']
    heatmap = frame_state['heatmap']
    untracked_pts = frame_state['untracked_pts']
    tracks_to_draw = frame_state['tracks']

    out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
    tracker.draw_tracks(out1, tracks_to_draw)
    if draw_keypoints and untracked_pts.shape[1] > 0:
      for pt in untracked_pts[:2, :].T:
        pt1 = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(out1, pt1, 1, (0, 0, 255), -1, lineType=16)
    if opt.show_extra:
      cv2.putText(out1, 'Point Tracks', font_pt, font, font_sc, font_clr, lineType=16)

    # Extra output -- Show current point detections.
    out2 = (np.dstack((img, img, img)) * 255.).astype('uint8')
    if (opt.show_extra or draw_keypoints) and pts.shape[1] > 0:
      for pt in pts[:2, :].T:
        pt1 = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(out2, pt1, 1, (0, 0, 255), -1, lineType=16)
    cv2.putText(out2, 'Raw Point Detections', font_pt, font, font_sc, font_clr, lineType=16)

    # Extra output -- Show the point confidence heatmap.
    if heatmap is not None:
      heatmap_vis = heatmap.copy()
      min_conf = 0.001
      heatmap_vis[heatmap_vis < min_conf] = min_conf
      heatmap_vis = -np.log(heatmap_vis)
      heatmap_vis = (heatmap_vis - heatmap_vis.min()) / (
          heatmap_vis.max() - heatmap_vis.min() + .00001)
      out3 = myjet[np.round(np.clip(heatmap_vis*10, 0, 9)).astype('int'), :]
      out3 = (out3*255).astype('uint8')
      if out3.shape[:2] != out2.shape[:2]:
        out3 = cv2.resize(out3, (out2.shape[1], out2.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    else:
      out3 = np.zeros_like(out2)
    cv2.putText(out3, 'Raw Point Confidences', font_pt, font, font_sc, font_clr, lineType=16)

    # Resize final output.
    if opt.show_extra:
      combined = np.hstack((out1, out2, out3))
      return cv2.resize(
          combined, (3*opt.display_scale*opt.W, opt.display_scale*opt.H))
    return cv2.resize(
        out1, (opt.display_scale*opt.W, opt.display_scale*opt.H))

  def run_superpoint_pass(current_img: np.ndarray,
                          baseline_state: Dict[str, Any],
                          frame_label: str) -> Tuple[Dict[str, Any], float]:
    """Run detector + tracker update starting from a saved tracker state."""
    if baseline_state and 'nn_thresh' in baseline_state:
      baseline_state['nn_thresh'] = fe.nn_thresh
    tracker.load_state(baseline_state)
    tracker.nn_thresh = fe.nn_thresh
    forward_start = time.time()
    pts, desc, heatmap = fe.run(current_img)
    forward_end = time.time()

    tracker.update(pts, desc)
    tracks = tracker.get_tracks(opt.min_length)
    render_mask = get_untracked_points_mask(pts, tracks, tracker)
    untracked_pts = pts[:, render_mask]

    raw_track_scores = tracks[:, 1].copy() if tracks.size != 0 else np.zeros((0,))
    track_lengths = np.sum(tracks[:, 2:] != -1, axis=1) if tracks.size != 0 else np.zeros((0,))
    track_ids = tracks[:, 0].astype(int) if tracks.size != 0 else np.zeros((0,), dtype=int)

    tracked_confidences = np.zeros((0,))
    tracked_scores = np.zeros((0,))
    if tracks.size != 0 and pts.shape[1] != 0:
      offsets = tracker.get_offsets()
      if offsets.size != 0:
        current_offset = offsets[-1]
        local_indices = tracks[:, -1].astype(int) - current_offset
        valid_mask = (local_indices >= 0) & (local_indices < pts.shape[1])
        if np.any(valid_mask):
          tracked_confidences = pts[2, local_indices[valid_mask]]
          tracked_scores = raw_track_scores[valid_mask]

    tracks_to_draw = tracks.copy()
    if tracks_to_draw.size != 0:
      tracks_to_draw[:, 1] /= float(fe.nn_thresh)

    updated_state: Dict[str, Any] = {
        'img': current_img,
        'pts': pts,
        'heatmap': heatmap,
        'untracked_pts': untracked_pts,
        'tracks': tracks_to_draw,
        'tracker_pre': baseline_state,
        'frame_name': frame_label,
        'track_scores_raw': raw_track_scores,
        'track_lengths': track_lengths,
        'track_ids': track_ids,
        'tracked_confidences': tracked_confidences,
        'tracked_scores': tracked_scores,
    }
    updated_state['tracker_post'] = tracker.save_state()
    return updated_state, float(forward_end - forward_start)

  def build_help_panel(frame_label: str,
                       step_mode_active: bool,
                       keypoints_visible: bool) -> np.ndarray:
    """Render the help summary and parameter dashboard."""
    panel_width = 360
    top_margin = 26
    line_h = 22
    bottom_margin = 20

    controls = [
        'Controls:',
        " q: quit    s: step",
        " ,: previous",
        " .: next",
        " k: toggle keypoints",
        " e/r: conf +/-",
        " d/f: NMS +/-",
        " t/g: match +/-",
    ]
    status = [
        '',
        'Status:',
        f" step mode: {'ON' if step_mode_active else 'OFF'}",
        f" keypoints: {'ON' if keypoints_visible else 'OFF'}",
        f' conf thresh: {fe.conf_thresh:.4f}',
        f' nms dist: {fe.nms_dist}',
        f' match thresh: {fe.nn_thresh:.2f}',
        f' frame: {frame_label}',
    ]
    if tracks_stale:
      status.append(' tracks view stale: replay to refresh')

    lines = controls + status
    panel_height = max(260, top_margin + bottom_margin + line_h * len(lines))
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    panel[:] = 24
    origin_y = top_margin
    for idx, text in enumerate(lines):
      cv2.putText(panel, text, (12, origin_y + idx*line_h),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=16)
    return panel

  def record_frame_metrics(frame_state: Dict[str, Any],
                           forward_time: float,
                           total_time: float) -> None:
    if not report_enabled:
      return
    frame_idx = len(report_rows)
    frame_label = frame_state['frame_name']
    pts = frame_state['pts']
    num_keypoints = int(pts.shape[1])
    confidences = pts[2, :] if num_keypoints > 0 else np.zeros((0,))
    num_tracks = int(frame_state['track_lengths'].size)
    track_lengths_arr = frame_state['track_lengths']
    track_scores_arr = frame_state['track_scores_raw']
    tracked_conf_arr = frame_state['tracked_confidences']
    tracked_scores_arr = frame_state['tracked_scores']
    untracked_pts = frame_state['untracked_pts']
    untracked_ratio = float(untracked_pts.shape[1]) / num_keypoints if num_keypoints > 0 else 0.0

    frame_metrics: Dict[str, Any] = {
        'frame_idx': frame_idx,
        'frame': frame_label,
        'num_keypoints': num_keypoints,
        'keypoint_conf_mean': float(confidences.mean()) if confidences.size else 0.0,
        'keypoint_conf_std': float(confidences.std()) if confidences.size else 0.0,
        'keypoint_conf_min': float(confidences.min()) if confidences.size else 0.0,
        'keypoint_conf_max': float(confidences.max()) if confidences.size else 0.0,
        'keypoint_conf_median': float(np.median(confidences)) if confidences.size else 0.0,
        'untracked_ratio': untracked_ratio,
        'num_tracks': num_tracks,
        'track_length_mean': float(track_lengths_arr.mean()) if track_lengths_arr.size else 0.0,
        'track_length_std': float(track_lengths_arr.std()) if track_lengths_arr.size else 0.0,
        'track_length_min': float(track_lengths_arr.min()) if track_lengths_arr.size else 0.0,
        'track_length_max': float(track_lengths_arr.max()) if track_lengths_arr.size else 0.0,
        'track_length_median': float(np.median(track_lengths_arr)) if track_lengths_arr.size else 0.0,
        'track_score_mean': float(track_scores_arr.mean()) if track_scores_arr.size else 0.0,
        'track_score_std': float(track_scores_arr.std()) if track_scores_arr.size else 0.0,
        'track_score_min': float(track_scores_arr.min()) if track_scores_arr.size else 0.0,
        'track_score_max': float(track_scores_arr.max()) if track_scores_arr.size else 0.0,
        'track_score_median': float(np.median(track_scores_arr)) if track_scores_arr.size else 0.0,
        'tracked_confidence_mean': float(tracked_conf_arr.mean()) if tracked_conf_arr.size else 0.0,
        'tracked_confidence_std': float(tracked_conf_arr.std()) if tracked_conf_arr.size else 0.0,
        'tracked_confidence_min': float(tracked_conf_arr.min()) if tracked_conf_arr.size else 0.0,
        'tracked_confidence_max': float(tracked_conf_arr.max()) if tracked_conf_arr.size else 0.0,
        'tracked_confidence_median': float(np.median(tracked_conf_arr)) if tracked_conf_arr.size else 0.0,
        'forward_time_ms': float(forward_time * 1000.0),
        'total_time_ms': float(total_time * 1000.0),
    }

    for length in range(1, opt.max_length + 1):
      count = int(np.sum(track_lengths_arr == length)) if track_lengths_arr.size else 0
      frame_metrics[f'tracks_len_{length}'] = count

    report_rows.append(frame_metrics)
    keypoint_counts.append(num_keypoints)
    keypoint_conf_all.extend(confidences.tolist())
    track_counts.append(num_tracks)
    track_lengths_all.extend(track_lengths_arr.astype(int).tolist())
    track_scores_all.extend(track_scores_arr.tolist())
    tracked_confidences_all.extend(tracked_conf_arr.tolist())
    tracked_scores_for_corr.extend(tracked_scores_arr.tolist())
    untracked_ratios.append(untracked_ratio)
    forward_times_ms.append(frame_metrics['forward_time_ms'])
    total_times_ms.append(frame_metrics['total_time_ms'])

    for track_id, length, score in zip(frame_state['track_ids'], track_lengths_arr, track_scores_arr):
      track_stats[int(track_id)] = {'length': float(length), 'score': float(score)}

    if report_enabled and report_total_frames and report_total_frames > 0:
      completed = frame_idx + 1
      fraction = min(1.0, completed / float(report_total_frames))
      last_fraction = progress_state['last_fraction']
      if fraction - last_fraction >= (1.0 / report_total_frames) or fraction >= 1.0:
        filled = int(round(fraction * progress_bar_width))
        bar = '#' * filled + '-' * (progress_bar_width - filled)
        sys.stdout.write(f'\rReport progress: [{bar}] {fraction*100:5.1f}% ({completed}/{report_total_frames})')
        sys.stdout.flush()
        progress_state['last_fraction'] = fraction
      if fraction >= 1.0:
        sys.stdout.write('\n')

  def _basic_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
      return {'mean': float('nan'), 'std': float('nan'), 'min': float('nan'),
              'max': float('nan'), 'median': float('nan')}
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'median': float(np.median(arr)),
    }

  def finalize_report() -> None:
    if not report_enabled:
      return
    if not report_rows:
      print('Report requested but no frames processed; skipping report generation.')
      return

    metrics_csv = report_dir / 'metrics.csv'
    length_fields = [f'tracks_len_{length}' for length in range(1, opt.max_length + 1)]
    csv_fields = ['frame_idx', 'frame', 'num_keypoints', 'keypoint_conf_mean',
                  'keypoint_conf_std', 'keypoint_conf_min', 'keypoint_conf_max',
                  'keypoint_conf_median', 'untracked_ratio', 'num_tracks',
                  'track_length_mean', 'track_length_std', 'track_length_min',
                  'track_length_max', 'track_length_median', 'track_score_mean',
                  'track_score_std', 'track_score_min', 'track_score_max',
                  'track_score_median', 'tracked_confidence_mean',
                  'tracked_confidence_std', 'tracked_confidence_min',
                  'tracked_confidence_max', 'tracked_confidence_median',
                  'forward_time_ms', 'total_time_ms'] + length_fields

    with metrics_csv.open('w', newline='') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
      writer.writeheader()
      for row in report_rows:
        writer.writerow(row)

    summary_csv = report_dir / 'summary.csv'
    summary_rows: List[Dict[str, Any]] = []
    summary_rows.append({'metric': 'num_keypoints', **_basic_stats(keypoint_counts)})
    summary_rows.append({'metric': 'num_tracks', **_basic_stats(track_counts)})
    summary_rows.append({'metric': 'track_length', **_basic_stats(track_lengths_all)})
    summary_rows.append({'metric': 'track_score', **_basic_stats(track_scores_all)})
    summary_rows.append({'metric': 'keypoint_confidence', **_basic_stats(keypoint_conf_all)})
    summary_rows.append({'metric': 'tracked_confidence', **_basic_stats(tracked_confidences_all)})
    summary_rows.append({'metric': 'untracked_ratio', **_basic_stats(untracked_ratios)})
    summary_rows.append({'metric': 'forward_time_ms', **_basic_stats(forward_times_ms)})
    summary_rows.append({'metric': 'total_time_ms', **_basic_stats(total_times_ms)})

    corr_value = float('nan')
    if len(tracked_confidences_all) > 1 and len(tracked_scores_for_corr) == len(tracked_confidences_all):
      corr_matrix = np.corrcoef(tracked_confidences_all, tracked_scores_for_corr)
      if corr_matrix.shape == (2, 2):
        corr_value = float(corr_matrix[0, 1])
    summary_rows.append({'metric': 'confidence_match_score_correlation',
                         'mean': corr_value})

    with summary_csv.open('w', newline='') as csvfile:
      fieldnames = ['metric', 'mean', 'std', 'min', 'max', 'median']
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      writer.writeheader()
      for row in summary_rows:
        writer.writerow(row)

    track_length_counts = Counter(int(stat['length']) for stat in track_stats.values())
    track_lengths_csv = report_dir / 'track_length_distribution.csv'
    with track_lengths_csv.open('w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(['length', 'count'])
      for length in sorted(track_length_counts):
        writer.writerow([length, track_length_counts[length]])

    track_scores_csv = report_dir / 'track_scores.csv'
    with track_scores_csv.open('w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(['track_id', 'average_score'])
      for track_id, stats in sorted(track_stats.items()):
        writer.writerow([track_id, stats['score']])

    try:
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as plt
    except ImportError:
      print('matplotlib not available; skipping plot generation.')
      return

    def _save_plot(fig, name: str) -> None:
      output_path = report_dir / name
      fig.savefig(output_path, bbox_inches='tight')
      plt.close(fig)

    frames = [row['frame_idx'] for row in report_rows]
    keypoints_per_frame = [row['num_keypoints'] for row in report_rows]
    if keypoints_per_frame:
      fig, ax = plt.subplots()
      ax.plot(frames, keypoints_per_frame, marker='o')
      ax.set_xlabel('Frame Index')
      ax.set_ylabel('Keypoints')
      ax.set_title('Keypoints per Frame')
      _save_plot(fig, 'keypoints_per_frame.png')

    if keypoint_conf_all:
      values = np.asarray(keypoint_conf_all)
      weights = np.ones_like(values, dtype=float) / values.size
      fig = plt.figure()
      gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
      ax_hist = fig.add_subplot(gs[0])
      ax_box = fig.add_subplot(gs[1], sharex=ax_hist)
      ax_hist.hist(values, bins=40, weights=weights, color='steelblue')
      ax_hist.set_ylabel('Fraction')
      ax_hist.set_title('Keypoint Confidence Distribution')
      ax_hist.text(0.98, 0.85, f'n={values.size}', transform=ax_hist.transAxes,
                   ha='right', va='top', fontsize=8, color='dimgray')
      ax_box.boxplot(values, vert=False, showmeans=True, patch_artist=True,
                     boxprops={'facecolor': 'lavender'}, meanprops={'color': 'firebrick'})
      ax_box.set_yticks([])
      ax_box.set_xlabel('Confidence')
      ax_box.grid(axis='x', linestyle='--', alpha=0.3)
      plt.setp(ax_hist.get_xticklabels(), visible=False)
      _save_plot(fig, 'keypoint_confidence_hist.png')

    if track_lengths_all:
      lengths = np.asarray(track_lengths_all)
      weights = np.ones_like(lengths, dtype=float) / lengths.size
      bins = range(1, int(max(lengths)) + 2)
      fig = plt.figure()
      gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
      ax_hist = fig.add_subplot(gs[0])
      ax_box = fig.add_subplot(gs[1], sharex=ax_hist)
      ax_hist.hist(lengths, bins=bins, weights=weights, align='left', rwidth=0.8)
      ax_hist.set_ylabel('Fraction')
      ax_hist.set_title('Track Length Distribution')
      ax_hist.text(0.98, 0.85, f'n={lengths.size}', transform=ax_hist.transAxes,
                   ha='right', va='top', fontsize=8, color='dimgray')
      ax_box.boxplot(lengths, vert=False, showmeans=True, patch_artist=True,
                     boxprops={'facecolor': 'mistyrose'}, meanprops={'color': 'firebrick'})
      ax_box.set_yticks([])
      ax_box.set_xlabel('Track Length')
      ax_box.grid(axis='x', linestyle='--', alpha=0.3)
      plt.setp(ax_hist.get_xticklabels(), visible=False)
      _save_plot(fig, 'track_length_hist.png')

    if track_scores_all:
      scores = np.asarray(track_scores_all)
      weights = np.ones_like(scores, dtype=float) / scores.size
      fig = plt.figure()
      gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
      ax_hist = fig.add_subplot(gs[0])
      ax_box = fig.add_subplot(gs[1], sharex=ax_hist)
      ax_hist.hist(scores, bins=40, weights=weights, color='seagreen')
      ax_hist.set_ylabel('Fraction')
      ax_hist.set_title('Track Match Score Distribution')
      ax_hist.text(0.98, 0.85, f'n={scores.size}', transform=ax_hist.transAxes,
                   ha='right', va='top', fontsize=8, color='dimgray')
      ax_box.boxplot(scores, vert=False, showmeans=True, patch_artist=True,
                     boxprops={'facecolor': 'honeydew'}, meanprops={'color': 'firebrick'})
      ax_box.set_yticks([])
      ax_box.set_xlabel('Average Match Score')
      ax_box.grid(axis='x', linestyle='--', alpha=0.3)
      plt.setp(ax_hist.get_xticklabels(), visible=False)
      _save_plot(fig, 'track_score_hist.png')

    if untracked_ratios:
      fig, ax = plt.subplots()
      ax.plot(frames, untracked_ratios, marker='x', color='darkorange')
      ax.set_xlabel('Frame Index')
      ax.set_ylabel('Untracked Ratio')
      ax.set_title('Untracked Keypoint Ratio per Frame')
      _save_plot(fig, 'untracked_ratio_per_frame.png')

    config_snapshot = {
        'run': {
            'timestamp': timestamp,
            'input': opt.input,
            'weights_path': opt.weights_path,
            'model': opt.model,
            'total_frames_expected': report_total_frames,
            'frames_processed': len(report_rows),
        },
        'thresholds': {
            'conf_thresh': opt.conf_thresh,
            'nms_dist': opt.nms_dist,
            'nn_thresh': opt.nn_thresh,
            'min_track_length': opt.min_length,
            'max_track_length': opt.max_length,
        },
        'report': {
            'enabled': report_enabled,
            'output_dir': str(report_dir),
            'metrics_csv': 'metrics.csv',
            'summary_csv': 'summary.csv',
            'track_length_distribution_csv': 'track_length_distribution.csv',
            'track_scores_csv': 'track_scores.csv',
        }
    }

    config_path = report_dir / 'report_config.yaml'
    config_path.write_text(yaml.dump(config_snapshot, sort_keys=True), encoding='utf-8')

  frame_state: Dict[str, Any] = {}
  frame_history: List[Dict[str, Any]] = []
  history_index = -1
  step_mode = not opt.write and not opt.no_display
  advance_requested = True
  redraw_requested = False
  tracks_stale = False

  print('==> Running Demo.')
  if not opt.no_display:
    print("Keyboard: 'q' to quit, 'k' toggle keypoints, 's' toggle step mode.")
    print("Keyboard: 'e/r' adjust confidence, 'd/f' adjust NMS, 't/g' adjust match threshold.")
    print("Step mode: ',' moves back, '.' advances.")
    if step_mode:
      print("Step mode is active. Use '.' to advance, ',' to revisit the previous frame.")
  while True:
    if advance_requested:
      start = time.time()
      img, status = vs.next_frame()
      if status is False:
        break

      baseline_state = tracker.save_state()
      frame_label = vs.last_name or 'frame_%06d' % max(vs.i - 1, 0)
      frame_state, forward_time = run_superpoint_pass(img, baseline_state, frame_label)
      if history_index < len(frame_history) - 1:
        frame_history[:] = frame_history[:history_index + 1]
      frame_history.append(frame_state)
      history_index = len(frame_history) - 1
      out = build_visualization(frame_state, show_keypoints)

      if opt.write:
        out_file = os.path.join(opt.write_dir, 'frame_%05d.png' % vs.i)
        print('Writing image to %s' % out_file)
        cv2.imwrite(out_file, out)

      end = time.time()
      safe_forward = max(forward_time, 1e-6)
      safe_total = max(end - start, 1e-6)
      net_t = 1. / safe_forward
      total_t = 1. / safe_total
      if opt.show_extra:
        print('Processed image %d (net+post_process: %.2f FPS, total: %.2f FPS).' %
              (vs.i, net_t, total_t))

      record_frame_metrics(frame_state, forward_time, end - start)

      advance_requested = not step_mode
      redraw_requested = False
    else:
      if not frame_state:
        advance_requested = True
        continue
      if redraw_requested and history_index >= 0:
        baseline_state = frame_state['tracker_pre']
        frame_state, _ = run_superpoint_pass(frame_state['img'], baseline_state,
                                             frame_state['frame_name'])
        frame_history[history_index] = frame_state
        redraw_requested = False
        advance_requested = not step_mode
      out = build_visualization(frame_state, show_keypoints)

    if not opt.no_display:
      cv2.imshow(win, out)
      current_frame_label = frame_state['frame_name'] if frame_state else '-'
      dashboard = build_help_panel(current_frame_label, step_mode, show_keypoints)
      cv2.imshow(info_win, dashboard)
      wait_time = 0 if step_mode else opt.waitkey
      key = cv2.waitKey(wait_time) & 0xFF
      if key == ord('q'):
        print('Quitting, \'q\' pressed.')
        break
      if key == ord('k'):
        show_keypoints = not show_keypoints
        state_msg = 'Showing' if show_keypoints else 'Hiding'
        print("{} untracked keypoints (toggle 'k').".format(state_msg))
      elif key == ord('s'):
        step_mode = not step_mode
        if step_mode:
          advance_requested = False
          print("Step mode enabled. Use '.' to advance, ',' to revisit the previous frame.")
        else:
          advance_requested = True
          print('Step mode disabled. Resuming continuous playback.')
      elif key == ord(',') and step_mode:
        if history_index > 0:
          history_index -= 1
          frame_state = frame_history[history_index]
          if 'tracker_pre' in frame_state:
            frame_state['tracker_pre']['nn_thresh'] = fe.nn_thresh
          if 'tracker_post' in frame_state:
            frame_state['tracker_post']['nn_thresh'] = fe.nn_thresh
            tracker.load_state(frame_state['tracker_post'])
          else:
            tracker.nn_thresh = fe.nn_thresh
          tracker.nn_thresh = fe.nn_thresh
          advance_requested = False
          redraw_requested = True
        else:
          print('Reached beginning of frame history.')
      elif key == ord('.') and step_mode:
        if history_index < len(frame_history) - 1:
          history_index += 1
          frame_state = frame_history[history_index]
          if 'tracker_pre' in frame_state:
            frame_state['tracker_pre']['nn_thresh'] = fe.nn_thresh
          if 'tracker_post' in frame_state:
            frame_state['tracker_post']['nn_thresh'] = fe.nn_thresh
            tracker.load_state(frame_state['tracker_post'])
          else:
            tracker.nn_thresh = fe.nn_thresh
          tracker.nn_thresh = fe.nn_thresh
          advance_requested = False
          redraw_requested = True
        else:
          advance_requested = True
      elif key in (ord('e'), ord('r')):
        if key == ord('e'):
          delta = -0.1
        else:
          delta = 0.1
        fe.conf_thresh = float(np.clip(fe.conf_thresh * (1.0 + delta), 0.0001, 1.0))
        opt.conf_thresh = fe.conf_thresh
        print('Confidence threshold set to {:.4f}'.format(fe.conf_thresh))
        if not tracks_stale:
          print('Tracks view uses cached history; replay from start to fully refresh.')
        tracks_stale = True
        if frame_state:
          redraw_requested = True
          advance_requested = False
      elif key in (ord('d'), ord('f')):
        delta = -1 if key == ord('d') else 1
        fe.nms_dist = int(np.clip(fe.nms_dist + delta, 1, 20))
        opt.nms_dist = fe.nms_dist
        print('NMS distance set to {}'.format(fe.nms_dist))
        if not tracks_stale:
          print('Tracks view uses cached history; replay from start to fully refresh.')
        tracks_stale = True
        if frame_state:
          redraw_requested = True
          advance_requested = False
      elif key in (ord('t'), ord('g')):
        delta = -0.05 if key == ord('t') else 0.05
        fe.nn_thresh = float(np.clip(fe.nn_thresh + delta, 0.05, 1.5))
        tracker.nn_thresh = fe.nn_thresh
        if frame_state and 'tracker_pre' in frame_state:
          # Ensure the cached tracker snapshot uses the updated threshold.
          frame_state['tracker_pre']['nn_thresh'] = fe.nn_thresh
        opt.nn_thresh = fe.nn_thresh
        print('Match threshold set to {:.2f}'.format(fe.nn_thresh))
        if not tracks_stale:
          print('Tracks view uses cached history; replay from start to fully refresh.')
        tracks_stale = True
        if frame_state:
          redraw_requested = True
          advance_requested = False


  # Close any remaining windows.
  cv2.destroyAllWindows()

  finalize_report()

  print('==> Finshed Demo.')
