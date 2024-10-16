#include "ReconWorker.hpp"

void ReconWorker::start() {
  bool m_shouldStop = false;
  while (!m_shouldStop) {
    m_buffer->consume([&](std::shared_ptr<BScanData<ArpamFloat>> &data) {
      if (data == nullptr) {
        m_shouldStop = true;
      } else {
        m_recontsructor.recon(*data);
        emit imagesReady(data);

        if (m_exportAll) {
          const auto exportDir =
              m_exportDir / fmt::format("{:03}", data->frameIdx);

          if (m_annotations == nullptr) {
            data->exportToFile(exportDir);
          } else {
            data->exportToFile(exportDir, m_annotations->getAnnotationForFrame(
                                              data->frameIdx));
          }
        }
      }
    });
  }
}
