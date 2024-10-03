#pragma once

#include <array>
#include <iostream>
#include <mutex>

// Represents a multistage pipeline circular buffer.
// A producer (stage 0) produces into the circular buffer,
// while the results are consumed by each subsequent stage of the pipeline
// sequentially

// Multi-stage ring buffer class template
template <typename T, int Stages, int BUFFER_SIZE = 5>
class MultiStageRingBuffer {
public:
  MultiStageRingBuffer() {
    buffer.resize(BUFFER_SIZE);

    for (auto &count : stageBufferCount) {
      count = 0;
    }
    pos.fill(0);
  }

  void produceOne(std::function<void(T &)> produceFunc) {
    std::unique_lock<std::mutex> lock(mtx);
    // Wait until there is space in the buffer
    cv.wait(lock, [this] { return stageBufferCount[0] < BUFFER_SIZE; });

    // Write to the buffer
    produceFunc(buffer[writePos]);
    std::cout << "Producer: Produced " << buffer[writePos] << " at position "
              << writePos << std::endl;

    // Update buffer write position and stage count
    writePos = (writePos + 1) % BUFFER_SIZE;
    stageBufferCount[0]++; // Producer added data for stage 1

    // Notify stage 1 that new data is available
    lock.unlock();
    cv.notify_all();
  }

  bool consumeOne(int stageId, std::function<void(T &)> processFunc) {
    std::unique_lock<std::mutex> lock(mtx);

    // Wait until there is data available for this stage
    cv.wait(lock, [this, stageId] {
      return stageBufferCount[stageId - 1] > 0 || finished;
    });

    if (stageBufferCount[stageId - 1] > 0) {
      // Process the data
      int &readPos = pos[stageId - 1];
      auto &buf = buffer[readPos];
      processFunc(buffer[readPos]);

      std::cout << "Consumer " << stageId << ": consumed " << buffer[readPos]
                << " at position " << readPos << "\n";

      // Update the read position and buffer count
      readPos = (readPos + 1) % BUFFER_SIZE;
      stageBufferCount[stageId - 1]--; // Data has been consumed by this stage

      // Notify the next stage (or producer if stage 1)
      if (stageId < Stages) {
        stageBufferCount[stageId]++; // Next stage has new data
      }

      lock.unlock();
      cv.notify_all(); // Wake up next stage or producer
      return true;
    }
    return false;
  }

private:
  std::vector<T> buffer; // Shared buffer
  std::array<std::atomic<int>, Stages>
      stageBufferCount;        // Tracks how many items each stage can process
  std::array<int, Stages> pos; // Positions for each stage
  int writePos{0};             // Producer write position

  std::mutex mtx;
  std::condition_variable cv;
  bool finished{false}; // Signals when the producer is done
};
