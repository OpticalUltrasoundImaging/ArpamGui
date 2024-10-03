#include <array>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>
#include <uspam/MultiStageBuffer.hpp>

TEST(MultiStageBuffer, Main) {
  const int stages = 3; // Number of stages in the pipeline

  // TODO test produceOne and consumeOne

  // Create the ring buffer with 3 stages
  MultiStageRingBuffer<int, stages> ringBuffer;

  std::array<int, 10> input;
  std::iota(input.begin(), input.end(), 0);

  // Producer function
  auto produceFunc = [&]() {
    for (auto v : input) {
      ringBuffer.produceOne([&](int &data) { data = v; });
    }
  };

  // Stage processing functions
  auto stage1Process = [](int &data) {
    data *= 2; // Stage 1 doubles the data
  };
  auto stage2Process = [](int &data) {
    data += 3; // Stage 2 adds 3 to the data
  };
  auto stage3Process = [](int &data) {
    data /= 2; // Stage 3 divides the data by 2
  };

  // Launch producer thread
  std::thread producerThread(produceFunc);
  // Launch consumer stage threads
  std::thread stage1Thread([&ringBuffer, stage1Process]() {
    while (ringBuffer.consumeOne(1, stage1Process)) {
    }
  });
  std::thread stage2Thread([&ringBuffer, stage2Process]() {
    while (ringBuffer.consumeOne(2, stage2Process)) {
    }
  });

  std::thread stage3Thread([&ringBuffer, stage3Process]() {
    while (ringBuffer.consumeOne(3, stage3Process)) {
    }
  });

  // Join all threads
  producerThread.join();
  stage1Thread.join();
  stage2Thread.join();
  stage3Thread.join();

  std::cout << "Processing complete!" << std::endl;
}