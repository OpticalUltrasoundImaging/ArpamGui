#pragma once

#include "Common.hpp"
#include "strConvUtils.hpp"
#include "uspam/ioParams.hpp"
#include "uspam/reconParams.hpp"
#include <QImage>
#include <QThreadPool>
#include <QtDebug>
#include <QtLogging>
#include <RFBuffer.hpp>
#include <armadillo>
#include <kfr/dsp.hpp>
#include <kfr/kfr.h>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <uspam/fft.hpp>
#include <uspam/imutil.hpp>
#include <uspam/timeit.hpp>
#include <uspam/uspam.hpp>
#include <utility>

namespace Recon {
namespace fs = std::filesystem;

/**
Supports CV_8UC4, CV_8UC3, CV_8UC1, CV_64FC1
Create copies for all.
 */
QImage cvMatToQImage(const cv::Mat &mat);

// template <uspam::Floating T>
using T = ArpamFloat;
std::tuple<float, float, float> procOne(const uspam::SystemParams &system,
                                        const uspam::recon::ReconParams &params,
                                        BScanData_<T> &data, bool flip,
                                        bool isPA = false);

class ImageWriteTask : public QRunnable {
  QImage img;
  QString fname;

public:
  ImageWriteTask(QImage img, QString fname)
      : img(std::move(img)), fname(std::move(fname)) {}
  void run() override { img.save(fname); }
};

void reconBScan(BScanData<ArpamFloat> &data,
                const uspam::recon::ReconParams2 &params,
                const uspam::io::IOParams &ioparams);

void saveImages(BScanData<ArpamFloat> &data, const fs::path &saveDir);

class Reconstructor {
public:
  Reconstructor() = default;
  Reconstructor(const uspam::recon::ReconParams2 &params,
                const uspam::io::IOParams &ioparams)
      : m_params(params), m_ioparams(ioparams) {}

  void recon(BScanData<ArpamFloat> &data) const {
    std::unique_lock<std::mutex> lock(m_mtx);
    reconBScan(data, m_params, m_ioparams);
  }

  void setParams(const uspam::recon::ReconParams2 &params,
                 const uspam::io::IOParams &ioparams) {
    std::unique_lock<std::mutex> lock(m_mtx);
    m_params = params;
    m_ioparams = ioparams;
  }

private:
  uspam::recon::ReconParams2 m_params{
      uspam::recon::ReconParams2::system2024v2GUI()};
  uspam::io::IOParams m_ioparams{uspam::io::IOParams::system2024v2GUI()};

  mutable std::mutex m_mtx;
};

} // namespace Recon