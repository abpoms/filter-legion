#include "util.h"
#include "libgcs.h"
#include <stdexcept>

#include <fstream>
#include <iostream>
#include <sstream>

using namespace VAE;
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

FILE* read_gcs_file(std::string key,
                    std::string bucket,
                    std::string path) {
  // Setup in value
  GoString key_str;
  key_str.p = (char*)key.c_str();
  key_str.n = key.size();

  GoString bucket_str;
  bucket_str.p = (char*)bucket.c_str();
  bucket_str.n = bucket.size();

  GoString path_str;
  path_str.p = (char*)path.c_str();
  path_str.n = path.size();

  char *fifo = (char*) gcs_read(key_str, bucket_str, path_str).data;

  FILE *fp = NULL;
  if ((fp = fopen(fifo, "rb")) == NULL)
    throw std::runtime_error("Cannot open " + path);

  return fp;
}

FILE* write_gcs_file(std::string key,
                     std::string bucket,
                     std::string path) {
  // Setup in value
  GoString key_str;
  key_str.p = (char*)key.c_str();
  key_str.n = key.size();

  GoString bucket_str;
  bucket_str.p = (char*)bucket.c_str();
  bucket_str.n = bucket.size();

  GoString path_str;
  path_str.p = (char*)path.c_str();
  path_str.n = path.size();

  char *fifo = (char*) gcs_write(key_str, bucket_str, path_str).data;

  FILE *fp = NULL;
  if ((fp = fopen(fifo, "wb")) == NULL)
    throw std::runtime_error("Cannot open " + path);

  return fp;
}

void close_gcs_write_file(FILE* fp, std::string path) {
  fclose(fp);

  GoString path_str;
  path_str.p = (char*)path.c_str();
  path_str.n = path.size();
  gcs_ensure_writes_are_done(path_str);
}

bool read_line(std::string &line, FILE *fp) {
  static char *line_buf = NULL;
  static size_t line_len = 0;

  __ssize_t string_len = getline(&line_buf, &line_len, fp);

  if (string_len == -1) {
    return false;
  } else {
    line = std::string(line_buf, string_len - 1);
    return true;
  }
}

void delete_file(const std::string &file) {
  std::ostringstream strm;
  strm << "/bin/rm -f " << file;
  system(strm.str().c_str());
}


bool get_raw_pointer(RegionAccessor<AccessorType::Generic, void> acc,
                     Rect<1> dom,
                     char **ptr,
                     size_t size) {
  Rect<1> subrect;
  ByteOffset elem_stride[1];
  *ptr =
    reinterpret_cast<char*>(acc.raw_rect_ptr<1>(dom, subrect, elem_stride));
  return !(!*ptr || (subrect != dom) ||
           !offsets_are_dense<1>(dom,elem_stride, size));
}

char* get_image_pointer(ImageAccessor accessor,
                        int width, int height, int channels) {
  rect<1> img_rect(point<1>(0), point<1>(width * height * channels - 1));
  char* image_ptr = nullptr;
  bool success = get_raw_pointer(accessor, img_rect, &image_ptr, sizeof(char));
  // we should always have a direct pointer to the image
  if (!success) assert(false);
  return image_ptr;
}
