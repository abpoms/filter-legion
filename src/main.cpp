#include "util.h"

#include "legion.h"
#include "default_mapper.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Constants and typedefs
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
enum TunableVariables {
  NODE_COUNT_VAR,
}

enum TaskID {
  MAIN_TASK_ID,
  INNER_TASK_ID,
  LOAD_TASK_ID,
  FILTER_TASK_ID,
  FEATURE_TASK_ID,
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

const size_t PATH_SIZE = 256;
const size_t VEC_DIM = 4096;

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Mapper
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class FilterMapper : public FilterMapper {
public:
  FilterMapper(Machine m, HighLevelRuntime *rt, Processor local)
    : DefaultMapper(m, rt, local) {}
public:
  virtual void select_task_options(Task *task) {
    DefaultMapper::select_task_options(task);

    auto id = task->task_id;
    if (id == MAIN_TASK_ID) {
    } else if (id == INNER_TASK_ID) {
    } else if (id == LOAD_TASK_ID) {
    } else if (id == FILTER_TASK_ID) {
    }
  }
};

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Legion Tasks
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
void feature_task(const Task* task,
                  const std::vector<PhysicalRegion>& regions,
                  Context ctx,
                  HighLevelRuntime* rt) {
}

bool filter_task(const Task* task,
                 const std::vector<PhysicalRegion>& regions,
                 Context ctx,
                 HighLevelRuntime* rt) {
}

void load_task(const Task* task,
               const std::vector<PhysicalRegion>& regions,
               Context ctx,
               HighLevelRuntime* rt) {
  PhysicalRegion image_region = regions[0];

  std::string path(task->args, task->argsize);

  char* image_ptr = get_image_pointer(image_region.get_field_accessor(DATA_ID),
                                      400, 225, 3);
  FILE* fp = read_gcs_file(gcs_key, gcs_bucket, path);

  // Read image from GCS
  std::vector<char> input;
  {
    input.resize(1024);
    while (true) {
      size_t num_read = fread(input.data() - 1024, 1, 1024, fp);
      if (num_read != 1024) {
        input.resize(input.size() - (1024 - num_read));
        break;
      }
      input.resize(input.size() + 1024);
    }
  }
  fclose(fp);
}


void inner_task(const Task* task,
                const std::vector<PhysicalRegion>& regions,
                Context ctx,
                HighLevelRuntime* rt) {
  LogicalRegion path_region = task->regions[0].region;
  LogicalRegion vector_logical_region = task->regions[1].region;
  PhysicalRegion path_pr = regions[0];

  StringAccessor path_acc = path_pr.get_field_accessor(PATH_ID);

  Rect<1> path_rect =
    rt->get_index_space_domain(ctx, path_logical_region.get_index_space())
    .get_rect<1>();

  for (GenericPointInRectIterator<1> itr(path_rect); itr; itr++) {
    DomainPoint p = DomainPoint::from_point<1>(itr.p);
    std::string path = read_string(path_acc, p);

    // We could create these at the top level once because we are working with
    // images all of the same size but we might want to work with images of
    // multiple sizes
    const int width = 400;
    const int height = 225;
    const int channels = 3;

    Rect<1> image_rect(Point<1>(0), Point<1>(width * height * channels));
    IndexSpace image_is =
      rt->create_index_space(ctx, Domain::from_rect<1>(image_rect));

    FieldSpace image_fs = rt->create_field_space(ctx);
    {
      FieldSpaceAllocator allocator = rt->create_field_allocator(image_fs);
      allocator.alloc_field(sizeof(char), DATA_ID);
    }

    LogicalRegion image_region =
      rt->create_logical_region(ctx, image_is, image_fs);

    ///////////////////////////////////////////////////////////////////////////
    /// Load image
    TaskLauncher load_launcher(LOAD_TASK_ID,
                               TaskArgument(path.data(), path.size()));
    load_launcher.add_region_requirement
      (RegionRequirement(image_region, WRITE_ONLY, EXCLUSIVE, image_region));
    load_launcher.add_field(0, DATA_ID);

    ///////////////////////////////////////////////////////////////////////////
    /// Check if image passes filter
    TaskLauncher filter_launcher(FILTER_TASK_ID, TaskArgument());
    load_launcher.add_region_requirement
      (RegionRequirement(image_region, READ_ONLY, EXCLUSIVE, image_region));
    load_launcher.add_field(0, DATA_ID);

    load_launcher.add_region_requirement
      (RegionRequirement(vector_region, WRITE_ONLY, EXCLUSIVE, vector_region));
    load_launcher.add_field(1, FILTER_ID);

    Future f = rt->execute_task(ctx, filter_launcher);
    Predicate filter_result = rt->create_predicate(ctx, f);

    ///////////////////////////////////////////////////////////////////////////
    /// Compute feature vector from image
    TaskLauncher vector_launcher(FEATURE_TASK_ID, TaskArgument(),
                                 filter_result);
    load_launcher.add_region_requirement
      (RegionRequirement(image_region, READ_ONLY, EXCLUSIVE, image_region));
    load_launcher.add_field(0, DATA_ID);

    load_launcher.add_region_requirement
      (RegionRequirement(vector_region, WRITE_ONLY, EXCLUSIVE, vector_region));
    load_launcher.add_field(1, VEC_ID);
  }
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
      if (path.size() - 1 > PATH_SIZE) {
        std::cerr << "Path longer than maximum path size("
                  << PATH_SIZE << "): " << path << std::endl;
        exit(1);
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
    FieldSpaceAllocator allocator = rt->create_field_allocator(ctx, fs);
    allocator.allocate_field(PATH_SIZE * sizeof(char), PATH_ID);
  }

  LogicalRegion path_region = rt->create_logical_region(ctx, is, fs);

  /////////////////////////////////////////////////////////////////////////////
  /// Fill in path region
  {
    RegionRequirement req(path_region, WRITE_ONLY, EXCLUSIVE, path_region);
    InlineLauncher launch(req);
    PhysicalRegion pr = rt->map_region(ctx, launcher);
    pr.wait_until_valid();

    StringAccessor path_acc = pr.get_field_accessor(PATH_ID);

    int i = 0;
    for (GenericPointInRectIterator<1> itr(rect); itr; itr++, i++) {
      DomainPoint p = DomainPoint::from_point<1>(itr.p);
      write_string(path_acc, p, paths[i]);
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
    FieldSpaceAllocator allocator = rt->create_field_allocator(ctx, fs);
    allocator.allocate_field(VEC_DIM * sizeof(float), VEC_ID);
    allocator.allocate_field(sizeof(int), FILTER_ID);
  }

  LogicalRegion vector_region =
    rt->create_logical_region(ctx, vector_is, vector_fs);

  /////////////////////////////////////////////////////////////////////////////
  /// Partition path and vector region
  // Partition each image into its own color
  Rect<1> color_rect(Point<1>(0), Point<1>(paths.size() - 1));
  Domain color_domain(Domain::from_rect<1>(color_rect));

  IndexPartition path_index_partition =
    rt->create_equal_partition(ctx, is, color_domain);
  IndexPartition vector_index_partition =
    rt->create_equal_partition(ctx, vector_is, color_domain);

  LogicalPartition path_partition =
    rt->get_logical_partition(ctx, path_region, path_index_partition);
  LogicalPartition vector_partition =
    rt->get_logical_partition(ctx, vector_region, vector_index_partition);


  /////////////////////////////////////////////////////////////////////////////
  /// Launch filter task
  ArgumentMap argmap;
  IndexLauncher launcher(INNER_TASK_ID, color_domain, TaskArgument(), argmap);

  launcher.add_region_requirement
    (RegionRequirement(path_partition, READ_ONLY, EXCLUSIVE, path_region));
  launcher.add_field(0, PATH_ID);

  launcher.add_region_requirement
    (RegionRequirement(vector_partition, WRITE_ONLY, EXCLUSIVE, vector_region));
  launcher.add_field(1, VEC_ID);
  launcher.add_field(1, FILTER_ID);

  FutureMap fm = rt->execute_index_space(ctx, launcher);

  /////////////////////////////////////////////////////////////////////////////
  /// Compact feature vectors

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
    (MAIN_TASK_ID, Processor::LOC_PROC, true, false);

  HighLevelRuntime::register_legion_task<inner_task>
    (INNER_TASK_ID, Processor::LOC_PROC, false, true);

  HighLevelRuntime::register_legion_task<load_task>
    (LOAD_TASK_ID, Processor::IO_PROC, true, false);

  HighLevelRuntime::register_legion_task<bool, filter_task>
    (FILTER_TASK_ID, Processor::LOC_PROC, true, false);

  HighLevelRuntime::register_legion_task<feature_task>
    (FEATURE_TASK_ID, Processor::LOC_PROC, true, false);

  HighLevelRuntime::set_registration_callback(mapper_registration);

  HighLevelRuntime::start(argc, argv);
}
