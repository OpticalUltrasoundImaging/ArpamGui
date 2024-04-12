#include <CLI/CLI.hpp>
#include <chrono>
#include <filesystem>
#include <indicators/progress_bar.hpp>
#include <iostream>

#include <fftconv.hpp>
#include <mutex>
#include <uspam/uspam.hpp>
#include <fftw3.h>

std::mutex fftw_mutex;

namespace fs = std::filesystem;

namespace io = uspam::io;
namespace recon = uspam::recon;

auto estimate_aline_background_from_file(const io::IOParams &ioparams,
                                         const fs::path &fname,
                                         int num_alines = 0) {
  const auto num_alines_all = ioparams.get_num_scans<uint16_t>(fname, 1);
  if (num_alines < 1) {
    num_alines = num_alines_all;
  } else {
    num_alines = std::min(num_alines, num_alines_all);
  }

  arma::vec background =
      arma::mean(arma::conv_to<arma::mat>::from(
                     ioparams.load_rf<uint16_t>(fname, 1, 1, num_alines)),
                 1);
  return background;
}

using namespace std::chrono;
struct TimeIt {
  TimeIt(std::string_view name)
      : name(name), start(high_resolution_clock::now()) {}
  ~TimeIt() {
    auto elapsed = high_resolution_clock::now() - start;
    auto ms = duration_cast<milliseconds>(elapsed).count();
    std::cout << name << " " << ms << " ms\n";
  }
  std::string_view name;
  steady_clock::time_point start;
};

/**
BType is the dtype stored in the binary file
*/
template <typename BType>
void cli_recon(const fs::path fname, int starti = 0, int nscans = 0,
               const fs::path savedir = "images") {
  if (!fs::create_directory(savedir) && !fs::exists(savedir)) {
    std::cerr << " Failed to create savedir " << savedir << "\n";
    return;
  }

  const auto ioparams = uspam::io::IOParams::system2024v1();
  const auto params = recon::ReconParams2::system2024v1();

  const auto num_scans_all = ioparams.get_num_scans<BType>(fname);
  if (num_scans_all < 1) {
    std::cerr << "No scans available in " << fname << "\n";
    return;
  } else {
    std::cout << "num_scans_all: " << num_scans_all << "\n";
  }

  if (nscans < 1 || nscans > num_scans_all - starti) {
    nscans = num_scans_all - starti;
  }

  // const auto background_aline =
  //     estimate_aline_background_from_file(ioparams, fname, 50000);

  // const auto background = ioparams.split_rf_PAUS(background_aline);

  // background_aline.save("background.bin", arma::raw_binary);
  // background.PA.save("PAbackground.bin", arma::raw_binary);

  // using namespace indicators;
  // ProgressBar bar{option::BarWidth{50},
  //                 option::Start{" ["},
  //                 option::Fill{"="},
  //                 option::Lead{">"},
  //                 option::Remainder{"-"},
  //                 option::End{"]"},
  //                 option::PrefixText{"Recon..."},
  //                 option::ForegroundColor{Color::yellow},
  //                 option::ShowElapsedTime{true},
  //                 option::ShowRemainingTime{true},
  //                 option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};

  // auto rfPair = ioparams.allocate_split_pair<double>();
  // auto rfLog = io::PAUSpair<double>::zeros_like(rfPair);

  const int endi = nscans + starti;
  for (int i = starti; i < endi; ++i) {
    const double pct = (double)(i - starti) / nscans;
    // bar.set_progress(pct);

    const auto start = high_resolution_clock::now();

    const bool flip{i % 2 != 0};

    const auto rf_ = ioparams.load_rf<BType>(fname, i);
    arma::mat rf = arma::conv_to<arma::mat>::from(rf_);

    // Background subtraction
    rf.each_col() -= arma::mean(rf, 1);

    auto rfPair = ioparams.split_rf_PAUS(rf);
    // rf_pair.US.each_col() -= background.US.col(0);

    auto rfLog = io::PAUSpair<double>::zeros_like(rfPair);
    {
      TimeIt timeit("reconOneScan");
      params.reconOneScan(rfPair, rfLog, flip);
    }

    // rfLog.US.save("USlog.bin", arma::raw_binary);
    // rfLog.PA.save("PAlog.bin", arma::raw_binary);

    // cv::Mat PArect = uspam::imutil::makeRectangular(rfLog.PA);
    // cv::Mat USrect = uspam::imutil::makeRectangular(rfLog.US);

    const cv::Mat PAradial = uspam::imutil::makeRadial(rfLog.PA);
    const cv::Mat USradial = uspam::imutil::makeRadial(rfLog.US);

    const auto elapsed = high_resolution_clock::now() - start;
    std::cout << duration_cast<milliseconds>(elapsed).count() << " ms\n";

    cv::imshow("tmp", USradial);
    cv::waitKey(1);
  }
}

int main(int argc, char **argv) {
  fftconv::use_fftw_mutex(&fftw_mutex);
  uspam::fft::use_fftw_mutex(&fftw_mutex);

  CLI::App app{"arpam - reconstruct images from binfiles"};

  std::string _binpath;
  std::string savedir{"images"};
  int starti = 0;
  int nscans = 0;

  // app.add_option("binpath", _binpath)->required();
  // app.add_option("-s,--start-i", starti, "Start at scan i (optional)");
  // app.add_option("-n,--nscans", nscans, "Number of scans (optional)");
  // app.add_option("--savedir", nscans, "Directory to save images");
  // CLI11_PARSE(app, argc, argv);

  fs::path binpath{_binpath};

  // if (!fs::exists(binpath)) {
  //   std::cerr << "Error: the file path does not exist: " << binpath << "\n";
  //   return 1;
  // }

  // starti = 75;
  // nscans = 1;
  binpath = "F:/tmp/20231025 ex vivo 249/152449PAUS.bin";

  std::cout << "binpath: " << binpath << "\n";
  std::cout << "nscans: " << nscans << "\n";

  cli_recon<uint16_t>(binpath, starti, nscans, savedir);

  return 0;
}
