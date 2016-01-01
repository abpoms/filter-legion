#include "common.h"
#include "compute_features.h"
#include "util.h"
#include "jpeg/JPEGReader.h"

#include "legion.h"
#include "default_mapper.h"
#include "realm/realm.h"

#include <fstream>

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Constants and typedefs
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
enum TunableVariables {
  NODE_COUNT_VAR,
};

enum TaskID {
  MAIN_TASK_ID,
  INNER_TASK_ID,
  LOAD_TASK_ID,
  FILTER_TASK_ID,
  FEATURE_TASK_ID,
  REPARTITION_TASK_ID,
  KNN_TASK_ID,
};

enum MetadataIDs {
  PATH_ID,
};

enum ImageIDs {
  DATA_ID,
};

enum VectorIDs {
  VEC_ID,
  FILTER_ID,
};

const size_t K = 5;
const size_t PATH_SIZE = 256;

const int IMAGE_WIDTH = 400;
const int IMAGE_HEIGHT = 225;
const int IMAGE_CHANNELS = 3;

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Mapper
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class FilterMapper : public DefaultMapper {
public:
  FilterMapper(Machine m, HighLevelRuntime *rt, Processor local)
    : DefaultMapper(m, rt, local) {}
public:
  virtual void select_task_options(Task *task) {
    DefaultMapper::select_task_options(task);

    auto id = task->task_id;
    if (id == MAIN_TASK_ID) {
    } else if (id == INNER_TASK_ID) {
      task->regions[0].virtual_map = true;
      //task->regions[1].virtual_map = true;
      task->regions[2].virtual_map = true;
      task->task_priority = 4;
    } else if (id == LOAD_TASK_ID) {
      task->task_priority = 2;
    } else if (id == FILTER_TASK_ID) {
      task->task_priority = 3;
    } else if (id == FEATURE_TASK_ID) {
      task->task_priority = 2;
    }
  }
};

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Legion Tasks
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

void knn_task(const Task* task,
              const std::vector<PhysicalRegion>& regions,
              Context ctx,
              HighLevelRuntime* rt) {
  printf("hi\n");
  fflush(stdout);
}

IndexPartition
repartition_task(const Task* task,
                 const std::vector<PhysicalRegion>& regions,
                 Context ctx,
                 HighLevelRuntime* rt) {
  PhysicalRegion filter_result_region = regions[0];
  IndexSpace filter_is = filter_result_region.get_logical_region()
    .get_index_space();


  RegionAccessor<AccessorType::Generic> filter_acc =
    filter_result_region.get_field_accessor(FILTER_ID);

  printf("before create ind\n");
  fflush(stdout);
  IndexPartition p = rt->create_index_partition(ctx, filter_is, filter_acc);
  printf("after create\n");
  fflush(stdout);
  return p;
}

struct FeatureArgs {
  int batch_size;
};

void feature_task(const Task* task,
                  const std::vector<PhysicalRegion>& regions,
                  Context ctx,
                  HighLevelRuntime* rt) {
  FeatureArgs* args = (FeatureArgs*)task->args;

  PhysicalRegion vector_region = regions[0];

  std::vector<Frame> frames;
  for (int i = 0; i < args->batch_size; ++i) {
    PhysicalRegion image_region = regions[i+1];
    char* image_ptr =
      get_image_pointer(image_region.get_field_accessor(DATA_ID),
                        IMAGE_WIDTH,
                        IMAGE_HEIGHT,
                        IMAGE_CHANNELS);
    Frame frame;
    frame.width = IMAGE_WIDTH;
    frame.height = IMAGE_HEIGHT;
    frame.channels = IMAGE_CHANNELS;
    frame.element_size = sizeof(char);
    frame.data = image_ptr;
    frames.push_back(frame);
  }

  char* vector_ptr =
    (get_array_pointer(vector_region.get_field_accessor(VEC_ID),
                       1,
                       VEC_DIM * sizeof(float)));

  map_pool5_features(frames, vector_ptr);
}

bool filter_task(const Task* task,
                 const std::vector<PhysicalRegion>& regions,
                 Context ctx,
                 HighLevelRuntime* rt) {
  LogicalRegion vector_logical_region = task->regions[1].region;
  PhysicalRegion image_region = regions[0];
  PhysicalRegion vector_region = regions[1];

  char* image_ptr = get_image_pointer(image_region.get_field_accessor(DATA_ID),
                                      IMAGE_WIDTH,
                                      IMAGE_HEIGHT,
                                      IMAGE_CHANNELS);

  RegionAccessor<AccessorType::Generic, int> filter_acc =
    vector_region.get_field_accessor(FILTER_ID).typeify<int>();

  // Filter by checking first bit of image
  IndexIterator itr(rt, ctx, vector_logical_region);
  if (*image_ptr % 2 == 0) {
    filter_acc.write(itr.next(), 0);
    return true;
  } else {
    filter_acc.write(itr.next(), -1);
    return false;
  }
}

void load_task(const Task* task,
               const std::vector<PhysicalRegion>& regions,
               Context ctx,
               HighLevelRuntime* rt) {
  PhysicalRegion path_region = regions[0];
  PhysicalRegion image_region = regions[1];

  IndexSpace path_is = path_region.get_logical_region().get_index_space();
  Realm::Domain::DomainPointIterator itr
    (rt->get_index_space_domain(ctx, path_is));

  StringAccessor path_acc = path_region.get_field_accessor(PATH_ID);
  std::string path =
    read_string<PATH_SIZE>(path_acc, itr.p);
  printf("%s\n", path.c_str());
  fflush(stdout);

  FILE* fp = read_gcs_file(gcs_key, gcs_bucket, path);

  // Read image from GCS
  std::vector<char> input;
  {
    input.resize(1024);
    size_t size_before = 0;
    while (true) {
      size_t num_read = fread(input.data() + size_before, 1, 1024, fp);
      if (num_read != 1024) {
        input.resize(size_before + num_read);
        break;
      }
      size_before = input.size();
      input.resize(input.size() + 1024);
    }
  }
  fclose(fp);

  char* image_ptr = get_image_pointer(image_region.get_field_accessor(DATA_ID),
                                      IMAGE_WIDTH,
                                      IMAGE_HEIGHT,
                                      IMAGE_CHANNELS);

  // Decode image into raw data
  JPEGReader reader;
  reader.header_mem((uint8_t*)input.data(), input.size());
  std::vector<uint8_t*> rows(reader.height(), NULL);
  for (size_t i = 0; i < reader.height(); ++i) {
    rows[i] = (uint8_t*)(image_ptr + IMAGE_WIDTH * IMAGE_CHANNELS * i);
  }
  reader.load(rows.begin());
}


void inner_task(const Task* task,
                const std::vector<PhysicalRegion>& regions,
                Context ctx,
                HighLevelRuntime* rt) {
  LogicalRegion path_logical_region = task->regions[0].region;
  LogicalRegion vector_filter_logical_region = task->regions[1].region;
  LogicalRegion vector_data_logical_region = task->regions[2].region;

  IndexSpace path_is = path_logical_region.get_index_space();
  IndexSpace vector_is = vector_filter_logical_region.get_index_space();

  const int BATCH_SIZE = 32;

  // Partition into sub regions of size 1
  Domain path_even_domain =
    Domain::from_rect<1>
    (Rect<1>(Point<1>(0),
             Point<1>(rt->get_index_space_domain(ctx, path_is)
                      .get_volume() - 1)));
  Domain vector_even_domain =
    Domain::from_rect<1>
    (Rect<1>(Point<1>(0),
             Point<1>(rt->get_index_space_domain(ctx, vector_is)
                      .get_volume() - 1)));
  IndexPartition path_even_partition =
    create_even_partition(rt, ctx, path_is, path_even_domain);
  IndexPartition vector_even_partition =
    create_even_partition(rt, ctx, vector_is, vector_even_domain);

  LogicalPartition path_even_filter_partition =
    rt->get_logical_partition(ctx, path_logical_region,
                              path_even_partition);
  LogicalPartition vector_even_filter_partition =
    rt->get_logical_partition(ctx, vector_filter_logical_region,
                              vector_even_partition);

  // Partition into batched sub regions
  Domain batched_domain;
  IndexPartition vector_batched_partition =
    create_batched_partition(rt, ctx, vector_is, BATCH_SIZE, batched_domain);

  // LogicalPartition batched_filter_partition =
  //   rt->get_logical_partition(ctx, vector_filter_logical_region,
  //                             vector_batched_partition);
  LogicalPartition batched_data_partition =
    rt->get_logical_partition(ctx, vector_data_logical_region,
                              vector_batched_partition);


  Realm::Domain::DomainPointIterator even_itr(vector_even_domain);
  for (Realm::Domain::DomainPointIterator batched_itr(batched_domain);
       batched_itr;
       batched_itr++) {
    int current_batch_size = 0;
    std::vector<LogicalRegion> images;
    for (current_batch_size = 0; current_batch_size < BATCH_SIZE;
         current_batch_size++) {
      if (!even_itr) break;

      // We could create these at the top level once because we are working with
      // images all of the same size but we might want to work with images of
      // multiple sizes
      Rect<1> image_rect(Point<1>(0), Point<1>(IMAGE_WIDTH *
                                               IMAGE_HEIGHT *
                                               IMAGE_CHANNELS));
      IndexSpace image_is =
        rt->create_index_space(ctx, Domain::from_rect<1>(image_rect));

      FieldSpace image_fs = rt->create_field_space(ctx);
      {
        FieldAllocator allocator = rt->create_field_allocator(ctx, image_fs);
        allocator.allocate_field(sizeof(char), DATA_ID);
      }

      LogicalRegion image_region =
        rt->create_logical_region(ctx, image_is, image_fs);
      images.push_back(image_region);

      //////////////////////////////////////////////////////////////////////////
      /// Load image
      LogicalRegion path_filter_subregion =
        rt->get_logical_subregion_by_color(ctx, path_even_filter_partition,
                                           even_itr.p);

      TaskLauncher load_launcher(LOAD_TASK_ID, TaskArgument());
      load_launcher.add_region_requirement
        (RegionRequirement(path_filter_subregion, READ_ONLY, EXCLUSIVE,
                           path_logical_region));
      load_launcher.add_field(0, PATH_ID);

      load_launcher.add_region_requirement
        (RegionRequirement(image_region, WRITE_ONLY, EXCLUSIVE, image_region));
      load_launcher.add_field(1, DATA_ID);

      rt->execute_task(ctx, load_launcher);

      //////////////////////////////////////////////////////////////////////////
      /// Check if image passes filter
      LogicalRegion vector_filter_subregion =
        rt->get_logical_subregion_by_color(ctx, vector_even_filter_partition,
                                           even_itr.p);

      TaskLauncher filter_launcher(FILTER_TASK_ID, TaskArgument());
      filter_launcher.add_region_requirement
        (RegionRequirement(image_region, READ_ONLY, EXCLUSIVE, image_region));
      filter_launcher.add_field(0, DATA_ID);

      filter_launcher.add_region_requirement
        (RegionRequirement(vector_filter_subregion,
                           WRITE_ONLY,
                           EXCLUSIVE,
                           vector_filter_logical_region));
      filter_launcher.add_field(1, FILTER_ID);

      rt->execute_task(ctx, filter_launcher);
      //Future f = rt->execute_task(ctx, filter_launcher);
      //Predicate filter_result = rt->create_predicate(ctx, f);

      even_itr++;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Compute feature vector from image
    LogicalRegion vector_data_subregion =
      rt->get_logical_subregion_by_color(ctx, batched_data_partition,
                                         batched_itr.p);

    FeatureArgs args;
    args.batch_size = current_batch_size;
    TaskLauncher vector_launcher(FEATURE_TASK_ID,
                                 TaskArgument(&args, sizeof(args)));

    vector_launcher.add_region_requirement
      (RegionRequirement(vector_data_subregion,
                         WRITE_ONLY,
                         EXCLUSIVE,
                         vector_data_logical_region));
    vector_launcher.add_field(0, VEC_ID);

    for (size_t i = 0; i < images.size(); ++i) {
      LogicalRegion image_region = images[i];

      vector_launcher.add_region_requirement
        (RegionRequirement(image_region, READ_ONLY, EXCLUSIVE, image_region));
      vector_launcher.add_field(i + 1, DATA_ID);
    }

    rt->execute_task(ctx, vector_launcher);

    for (size_t i = 0; i < images.size(); ++i) {
      LogicalRegion image_region = images[i];
      rt->destroy_index_space(ctx, image_region.get_index_space());
      rt->destroy_field_space(ctx, image_region.get_field_space());
      rt->destroy_logical_region(ctx, image_region);
    }
  }

  rt->destroy_index_partition(ctx, path_even_partition);
  rt->destroy_index_partition(ctx, vector_even_partition);
  rt->destroy_index_partition(ctx, vector_batched_partition);
}

void main_task(const Task* task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx,
               HighLevelRuntime *rt) {
  /////////////////////////////////////////////////////////////////////////////
  /// Load paths from file
  std::vector<std::string> paths;
  {
    std::ifstream image_paths_stream("images.txt");
    while (image_paths_stream.good()) {
      std::string path;
      std::getline(image_paths_stream, path);
      // -1 for null character
      if (path.size() > PATH_SIZE - 1) {
        std::cerr << "Path longer than maximum path size("
                  << PATH_SIZE << "): " << path << std::endl;
        exit(1);
      }
      if (path.size() == 0) {
        break;
      }
      paths.push_back(path);
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  /// Create path region
  Rect<1> rect(Point<1>(0), Point<1>(paths.size() - 1));
  IndexSpace is = rt->create_index_space(ctx, Domain::from_rect<1>(rect));

  FieldSpace fs = rt->create_field_space(ctx);
  {
    FieldAllocator allocator = rt->create_field_allocator(ctx, fs);
    allocator.allocate_field(PATH_SIZE * sizeof(char), PATH_ID);
  }

  LogicalRegion path_region = rt->create_logical_region(ctx, is, fs);

  /////////////////////////////////////////////////////////////////////////////
  /// Fill in path region
  {
    RegionRequirement req(path_region, WRITE_ONLY, EXCLUSIVE, path_region);
    req.add_field(PATH_ID);
    InlineLauncher launcher(req);
    PhysicalRegion pr = rt->map_region(ctx, launcher);
    pr.wait_until_valid();

    StringAccessor path_acc = pr.get_field_accessor(PATH_ID);

    int i = 0;
    for (GenericPointInRectIterator<1> itr(rect); itr; itr++, i++) {
      DomainPoint p = DomainPoint::from_point<1>(itr.p);
      write_string<PATH_SIZE>(path_acc, p, paths[i]);
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  /// Create vector region
  IndexSpace vector_is = rt->create_index_space(ctx, paths.size());
  {
    IndexAllocator allocator = rt->create_index_allocator(ctx, vector_is);
    allocator.alloc(paths.size());
  }

  FieldSpace vector_fs = rt->create_field_space(ctx);
  {
    FieldAllocator allocator = rt->create_field_allocator(ctx, vector_fs);
    allocator.allocate_field(VEC_DIM * sizeof(float), VEC_ID);
    allocator.allocate_field(sizeof(int), FILTER_ID);
  }

  LogicalRegion vector_region =
    rt->create_logical_region(ctx, vector_is, vector_fs);

  /////////////////////////////////////////////////////////////////////////////
  /// Partition path and vector region
  // Partition each image into its own color
  Rect<1> color_rect(Point<1>(0), Point<1>(1));
  Domain color_domain(Domain::from_rect<1>(color_rect));

  // Not implemented in non-shared low level runtime
  // IndexPartition path_index_partition =
  //   rt->create_equal_partition(ctx, is, color_domain);
  // IndexPartition vector_index_partition =
  //   rt->create_equal_partition(ctx, vector_is, color_domain);

  IndexPartition path_index_partition =
    create_even_partition(rt, ctx, is, color_domain);
  IndexPartition vector_index_partition =
    create_even_partition(rt, ctx, vector_is, color_domain);

  LogicalPartition path_partition =
    rt->get_logical_partition(ctx, path_region, path_index_partition);
  LogicalPartition vector_partition =
    rt->get_logical_partition(ctx, vector_region, vector_index_partition);


  /////////////////////////////////////////////////////////////////////////////
  /// Launch filter task
  ArgumentMap argmap;
  IndexLauncher launcher(INNER_TASK_ID, color_domain, TaskArgument(), argmap);

  launcher.add_region_requirement
    (RegionRequirement(path_partition, 0, READ_ONLY, EXCLUSIVE, path_region));
  launcher.add_field(0, PATH_ID);

  launcher.add_region_requirement
    (RegionRequirement(vector_partition, 0, WRITE_ONLY, EXCLUSIVE,
                       vector_region));
  launcher.add_field(1, FILTER_ID);

  launcher.add_region_requirement
    (RegionRequirement(vector_partition, 0, WRITE_ONLY, EXCLUSIVE,
                       vector_region));
  launcher.add_field(2, VEC_ID);

  FutureMap fm = rt->execute_index_space(ctx, launcher);

  /////////////////////////////////////////////////////////////////////////////
  /// Compact feature vectors
  TaskLauncher partition_launcher(REPARTITION_TASK_ID, TaskArgument());

  partition_launcher.add_region_requirement
    (RegionRequirement(vector_region, READ_ONLY, EXCLUSIVE, vector_region));
  partition_launcher.add_field(0, FILTER_ID);

  Future f = rt->execute_task(ctx, partition_launcher);

  IndexPartition filtered_partition = f.get_result<IndexPartition>();

  LogicalPartition vector_filtered_lp =
    rt->get_logical_partition(ctx, vector_region, filtered_partition);

  /////////////////////////////////////////////////////////////////////////////
  /// Create knn region based on filtered size
  IndexSpace filtered_is = rt->get_index_subspace(ctx, filtered_partition, 0);
  size_t filtered_size =
    rt->get_index_space_domain(ctx, filtered_is).get_volume();
  printf("input size: %lu, filtered size %lu\n", paths.size(), filtered_size);
  fflush(stdout);

  IndexSpace knn_is = rt->create_index_space(ctx, filtered_size);
  {
    IndexAllocator allocator = rt->create_index_allocator(ctx, knn_is);
    allocator.alloc(filtered_size);
  }

  FieldSpace knn_fs = rt->create_field_space(ctx);
  {
    FieldAllocator allocator = rt->create_field_allocator(ctx, knn_fs);
    allocator.allocate_field(K * (sizeof(float) + sizeof(int)), DATA_ID);
  }

  LogicalRegion knn_region = rt->create_logical_region(ctx, knn_is, knn_fs);

  /////////////////////////////////////////////////////////////////////////////
  /// Run KNN
  LogicalRegion vector_subregion =
    rt->get_logical_subregion_by_color(ctx, vector_filtered_lp, 0);

  TaskLauncher knn_launcher(KNN_TASK_ID, TaskArgument());

  knn_launcher.add_region_requirement
    (RegionRequirement(vector_subregion, READ_ONLY, EXCLUSIVE, vector_region));
  knn_launcher.add_field(0, VEC_ID);

  knn_launcher.add_region_requirement
    (RegionRequirement(knn_region, READ_WRITE, EXCLUSIVE, knn_region));
  knn_launcher.add_field(1, DATA_ID);

  rt->execute_task(ctx, knn_launcher);

  /////////////////////////////////////////////////////////////////////////////
  /// Cleanup

  rt->destroy_logical_region(ctx, path_region);
  rt->destroy_logical_region(ctx, vector_region);
  rt->destroy_logical_region(ctx, knn_region);

  rt->destroy_index_space(ctx, is);
  rt->destroy_index_space(ctx, vector_is);
  rt->destroy_index_space(ctx, knn_is);

  rt->destroy_field_space(ctx, fs);
  rt->destroy_field_space(ctx, vector_fs);
  rt->destroy_field_space(ctx, knn_fs);
}

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Startup
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

void mapper_registration(Machine m, HighLevelRuntime *rt,
                         const std::set<Processor> &local_procs) {
  for (std::set<Processor>::const_iterator it = local_procs.begin();
       it != local_procs.end(); it++) {
    rt->replace_default_mapper(new FilterMapper(m, rt, *it), *it);
  }
}


int main(int argc, char **argv) {
  HighLevelRuntime::set_top_level_task_id(MAIN_TASK_ID);
  HighLevelRuntime::register_legion_task<main_task>
    (MAIN_TASK_ID, Processor::LOC_PROC, true, false,
     AUTO_GENERATE_ID, TaskConfigOptions(),
     "main task");

  HighLevelRuntime::register_legion_task<inner_task>
    (INNER_TASK_ID, Processor::LOC_PROC, false, true,
     AUTO_GENERATE_ID, TaskConfigOptions(false, true/*inner task*/),
     "inner task");

  HighLevelRuntime::register_legion_task<load_task>
    (LOAD_TASK_ID, Processor::IO_PROC, true, true,
     AUTO_GENERATE_ID, TaskConfigOptions(false/*leaf task*/),
     "load task");

  HighLevelRuntime::register_legion_task<bool, filter_task>
    (FILTER_TASK_ID, Processor::LOC_PROC, true, true,
     AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf task*/),
     "filter task");

  HighLevelRuntime::register_legion_task<feature_task>
    (FEATURE_TASK_ID, Processor::LOC_PROC, true, true,
     AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf task*/),
     "feature task");

  HighLevelRuntime::register_legion_task<IndexPartition, repartition_task>
    (REPARTITION_TASK_ID, Processor::LOC_PROC, true, true,
     AUTO_GENERATE_ID, TaskConfigOptions(),
     "repartition task");

  HighLevelRuntime::register_legion_task<knn_task>
    (KNN_TASK_ID, Processor::LOC_PROC, true, true,
     AUTO_GENERATE_ID, TaskConfigOptions(),
     "knn task");

  HighLevelRuntime::set_registration_callback(mapper_registration);

  HighLevelRuntime::start(argc, argv);
}
