#ifndef UTIL_H_
#define UTIL_H_

#include <string>
#include <cstdio>

#include "legion.h"

using StringAccessor =
  LegionRuntime::Accessor::RegionAccessor
  <LegionRuntime::Accessor::AccessorType::Generic, void>;

using ImageAccessor =
  LegionRuntime::Accessor::RegionAccessor
  <LegionRuntime::Accessor::AccessorType::Generic, void>;

const std::string gcs_key = "keys/visualdb-12f1f722b05e.json";
const std::string gcs_bucket = "vdb-imagenet";

FILE* read_gcs_file(std::string key, std::string bucket, std::string path);

FILE* write_gcs_file(std::string key, std::string bucket, std::string path);

void close_gcs_write_file(FILE* fp, std::string path);

bool read_line(std::string &line, FILE *fp);

void delete_file(const std::string &file);

template<unsigned DIM>
static inline bool offsets_are_dense
  (const LegionRuntime::Arrays::Rect<DIM> &bounds,
   const LegionRuntime::Accessor::ByteOffset *offset,
   size_t size) {
  off_t exp_offset = size;
  for (unsigned i = 0; i < DIM; i++) {
    bool found = false;
    for (unsigned j = 0; j < DIM; j++) {
      if (offset[j].offset == exp_offset) {
        found = true;
        exp_offset *= (bounds.hi[j] - bounds.lo[j] + 1);
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}

bool get_raw_pointer(LegionRuntime::Accessor::RegionAccessor
                     <LegionRuntime::Accessor::AccessorType::Generic, void> acc,
                     LegionRuntime::Arrays::Rect<1> dom,
                     char **ptr,
                     size_t size);

template <int LENGTH>
std::string read_string(StringAccessor accessor,
                        LegionRuntime::DomainPoint point) {
  char buffer[LENGTH];
  accessor.read_untyped(point, buffer, LENGTH);
  // Performing this copy because string buffer isn't guaranteed to be
  // contiguous
  return std::move(std::string(buffer));
}

template <int LENGTH>
void write_string(StringAccessor accessor,
                  LegionRuntime::DomainPoint point,
                  const std::string& string) {
  accessor.write_untyped(point, string.c_str(), string.size() + 1);
}

char* get_image_pointer(ImageAccessor accessor,
                        int width, int height, int channels);

#endif // UTIL_H_
