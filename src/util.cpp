#include "util.h"
#include "libgcs.h"

#include "realm/realm.h"
#include <stdexcept>

#include <fstream>
#include <iostream>
#include <sstream>

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

char* get_array_pointer(UntypedAccessor accessor,
                        size_t extent,
                        size_t element_size) {
  Rect<1> array_rect(Point<1>(0), Point<1>(extent - 1));
  char* array_ptr = nullptr;
  bool success =
    get_raw_pointer(accessor, array_rect, &array_ptr, element_size);
  // we should always have a direct pointer to the data
  if (!success) assert(false);
  return array_ptr;
}

char* get_image_pointer(ImageAccessor accessor,
                        int width, int height, int channels) {
  return get_array_pointer(accessor, width * height * channels, sizeof(char));
}

IndexPartition create_even_partition(HighLevelRuntime* rt,
                                     Context ctx,
                                     IndexSpace is,
                                     Domain color_dom) {
  Domain index_domain = rt->get_index_space_domain(ctx, is);
  const size_t index_volume = index_domain.get_volume();

  const size_t color_volume = color_dom.get_volume();

  if (index_domain.get_dim() == 0) {
    PointColoring coloring;
    size_t elements_allocated = 0;
    IndexIterator is_itr(rt, ctx, is);

    size_t i = 0;
    for (Realm::Domain::DomainPointIterator itr(color_dom); itr; itr++) {
      DomainPoint color = itr.p;

      size_t elements =
        ceil(static_cast<double>(index_volume - elements_allocated) /
             (color_volume - i));
      for (size_t i = 0; i < elements; ++i) {
        if (is_itr.has_next()) {
          coloring[color].points.insert(is_itr.next());
        } else {
          assert(false);
        }
      }
      elements_allocated += elements;
      i++;
    }
    return rt->create_index_partition(ctx, is, color_dom, coloring);
  } else {
    DomainPointColoring coloring;
    size_t elements_allocated = 0;
    size_t i = 0;
    for (Realm::Domain::DomainPointIterator itr(color_dom); itr; itr++) {
      DomainPoint color = itr.p;

      size_t elements =
        ceil(static_cast<double>(index_volume - elements_allocated) /
             (color_volume - i));
      coloring[color] =
        Domain::from_rect<1>
        (Rect<1>(Point<1>(elements_allocated),
                 Point<1>(elements_allocated + elements - 1)));
      elements_allocated += elements;
      i++;
    }
    return rt->create_index_partition(ctx, is, color_dom, coloring);
  }
}

IndexPartition create_batched_partition(HighLevelRuntime* rt,
                                        Context ctx,
                                        IndexSpace is,
                                        int batch_size,
                                        Domain& color_dom) {
  Domain index_domain = rt->get_index_space_domain(ctx, is);
  const size_t index_volume = index_domain.get_volume();

  const size_t color_volume = index_volume / batch_size;
  Rect<1> color_rect = Rect<1>(Point<1>(0), Point<1>(color_volume - 1));
  color_dom = Domain::from_rect<1>(color_rect);

  if (index_domain.get_dim() == 0) {
    PointColoring coloring;
    size_t elements_allocated = 0;
    IndexIterator is_itr(rt, ctx, is);
    size_t i = 0;

    for (Realm::Domain::DomainPointIterator itr(color_dom); itr; itr++) {
      DomainPoint color = itr.p;

      size_t elements = batch_size;
      if (elements_allocated + elements > index_volume)
        elements = index_volume - elements_allocated;

      for (size_t i = 0; i < elements; ++i) {
        if (is_itr.has_next()) {
          coloring[color].points.insert(is_itr.next());
        } else {
          assert(false);
        }
      }

      elements_allocated += elements;
      i++;
    }
    return rt->create_index_partition(ctx, is, color_dom, coloring);
  } else {
    DomainPointColoring coloring;
    size_t elements_allocated = 0;
    size_t i = 0;
    for (Realm::Domain::DomainPointIterator itr(color_dom); itr; itr++) {
      DomainPoint color = itr.p;

      size_t elements = batch_size;
      if (elements_allocated + elements > index_volume)
        elements = index_volume - elements_allocated;

      coloring[color] =
        Domain::from_rect<1>
        (Rect<1>(Point<1>(elements_allocated),
                 Point<1>(elements_allocated + elements - 1)));
      elements_allocated += elements;
      i++;
    }
    return rt->create_index_partition(ctx, is, color_dom, coloring);
  }
}
