/**
 * Storage class to support binding to a lvalue or a rvalue
 * C++17
 *
 * https://www.fluentcpp.com/2022/05/16/how-to-store-an-lvalue-or-an-rvalue-in-the-same-object/
 */
#pragma once
#include <stdexcept>
#include <variant>

namespace lvalue_rvalue_storage {

template <typename T> struct ConstReference {
  T const &value_; // NOLINT
  explicit ConstReference(T const &value) : value_(value) {}
};

template <typename T> struct NonConstReference {
  T &value_; // NOLINT
  explicit NonConstReference(T &value) : value_(value) {}
};

template <typename T> struct Value {
  T value_;
  explicit Value(T &&value) : value_(std::move(value)) {}
};

// The storage class that accepts a lvalue or a rvalue
template <typename T>
using _Storage = std::variant<Value<T>, ConstReference<T>, NonConstReference<T>>;

// Use std::visit and the `overload` pattern
template <typename... Functions> struct overload : Functions... {
  using Functions::operator()...;
  explicit overload(Functions... functions) : Functions(functions)... {}
};

// Const access
template <typename T> T const &getConstReference(_Storage<T> const &storage) {
  return std::visit(
      overload([](Value<T> const &value) -> T const & { return value.value_; },
               [](NonConstReference<T> const &value) -> T const & {
                 return value.value_;
               },
               [](ConstReference<T> const &value) -> T const & {
                 return value.value_;
               }));
};

template <typename T> T &getReference(_Storage<T> &storage) {
  return std::visit(
      overload([](Value<T> &value) -> T & { return value.value_; },
               [](NonConstReference<T> &value) -> T & { return value.value_; },
               [](ConstReference<T> &) -> T & {
                 throw std::runtime_error{
                     "Cannot get a non const reference from a const reference"};
               }),
      storage);
}

} // namespace lvalue_rvalue_storage
