#pragma once

#include "Annotation/AnnotationJsonFile.hpp"
#include "Common.hpp"
#include "RFBuffer.hpp"
#include "Recon.hpp"
#include <QDebug>
#include <QObject>
#include <QtLogging>
#include <atomic>
#include <filesystem>
#include <fmt/base.h>
#include <utility>

class ReconWorker : public QObject {
  Q_OBJECT
public:
  explicit ReconWorker(std::shared_ptr<RFBuffer<ArpamFloat>> buffer)
      : m_buffer(std::move(buffer)) {}

  auto &reconstructor() { return m_recontsructor; }
  auto &reconstructor() const { return m_recontsructor; }

  /*
  Start main loop
  */
  void start();

  // Signal the worker to export frames, and give relevate states
  void shouldExportFrames(const fs::path &exportDir,
                          annotation::AnnotationJsonFile *annotations) {
    m_exportDir = exportDir;
    m_annotations = annotations;
    m_exportAll = true;
  }

  // Signal the worker to stop exporting frames
  void stopExportingFrames() { m_exportAll = false; }

signals:
  void imagesReady(std::shared_ptr<BScanData<ArpamFloat>> data);

private:
  std::shared_ptr<RFBuffer<ArpamFloat>> m_buffer;

  Recon::Reconstructor m_recontsructor;

  /*
  States related to the export all feature
  */
  // Should export when processing
  std::atomic<bool> m_exportAll{false};
  // Export root directory
  fs::path m_exportDir;
  // Annotations
  annotation::AnnotationJsonFile *m_annotations;
};