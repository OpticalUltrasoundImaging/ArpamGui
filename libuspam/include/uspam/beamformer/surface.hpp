#pragma once

#include <algorithm> // for std::fill
#include <armadillo>
#include <concepts>
#include <iostream>

namespace uspam::surface {

// Helper function to compute linear interpolation between two values using
// Armadillo
template <typename T>
auto linearInterpolate(T v1, T v2, int n)
  requires(std::is_floating_point_v<T>)
{
  // Create a linearly spaced vector between v1 and v2
  arma::Col<T> result = arma::linspace<arma::Col<T>>(
      v1, v2, n + 2); // n + 2 points including v1 and v2
  return result.subvec(
      1, n); // We return only the intermediate points (exclude v1 and v2)
}

// Helper function to check if the current segment is disjoint based on
// max_distance
template <typename T>
bool prevDisjoint(const arma::Col<T> &idx, int i_prev, int i_curr,
                  int max_distance) {
  return std::abs(idx[i_curr] - idx[i_prev]) > max_distance;
}

// Interpolation for the next gap
template <typename T> void interpNextGapInplace(arma::Col<T> &idx, int &i) {
  const int n = idx.n_elem;
  while (i < n) {
    if (idx[i] == 0) {
      int i_start = i;

      // Find the end of the gap
      while (i < n && idx[i] == 0) {
        i++;
      }
      const int i_end = i;

      // Interpolate the gap if it's in the middle
      if (i_start > 0 && i_end < n) {
        const int gap_size = i_end - i_start;
        const auto v1 = idx[i_start - 1];
        const auto v2 = idx[i_end];
        // Replace the gap with interpolated values
        idx.subvec(i_start, i_end - 1) = linearInterpolate(v1, v2, gap_size);
      }
    } else {
      i++;
    }
  }
}

// Fix surface index function
template <typename T> arma::Col<T> fixSurfaceIdxMissing(arma::Col<T> &idx) {
  const int MAX_DISTANCE = 30;
  int n = idx.n_elem;
  int i = 1;
  int last_good_i = 0; // Assume idx[0] is correct

  // Clear disjoint surfaces
  while (i < n) {
    // Skip over zeros
    while (i < n && idx[i] == 0) {
      i++;
    }

    if (i < n) {
      if (prevDisjoint(idx, last_good_i, i, MAX_DISTANCE + i - last_good_i)) {
        int disjoint_start = i;
        i++;
        while (i < n &&
               !prevDisjoint(idx, i - 1, i, MAX_DISTANCE + i - last_good_i)) {
          i++;
        }
        int disjoint_end = i;
        idx.subvec(disjoint_start, disjoint_end - 1).zeros();
      } else {
        last_good_i = i;
        i++;
      }
    }
  }

  // Handle interpolation in gaps (Case 2 first)
  i = 0;
  while (i < n && idx[i] == 0) {
    i++;
  }

  while (i < n) {
    interpNextGapInplace(idx, i);
  }

  // Handle Case 1 and Case 3 (zeros at start or end)
  if (idx[0] == 0 || idx[n - 1] == 0) {
    // Find the last non-zero element
    i = n - 1;
    while (i >= 0 && idx[i] == 0) {
      i--;
    }

    if (i < 0) {
      // All indices are zero, no surface found
      return idx; // Return as is
    }

    // Roll the vector
    int n_pts_rotate = n - i;
    idx = arma::shift(idx, n_pts_rotate);

    // Interpolate the first gap after rotation
    int temp_i = 1;
    interpNextGapInplace(idx, temp_i);

    // Rotate back
    idx = arma::shift(idx, -n_pts_rotate);
  }

  return idx;
}

} // namespace uspam::surface