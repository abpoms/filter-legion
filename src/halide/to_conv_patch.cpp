#include "Halide.h"
#include <stdio.h>
#include <algorithm>

/* Taken directly from Halide resize app */

using namespace Halide;

enum InterpolationType {
    BOX, LINEAR, CUBIC, LANCZOS
};

Expr kernel_box(Expr x) {
    Expr xx = abs(x);
    return select(xx <= 0.5f, 1.0f, 0.0f);
}

Expr kernel_linear(Expr x) {
    Expr xx = abs(x);
    return select(xx < 1.0f, 1.0f - xx, 0.0f);
}

Expr kernel_cubic(Expr x) {
    Expr xx = abs(x);
    Expr xx2 = xx * xx;
    Expr xx3 = xx2 * xx;
    float a = -0.5f;

    return select(xx < 1.0f, (a + 2.0f) * xx3 - (a + 3.0f) * xx2 + 1,
                  select (xx < 2.0f, a * xx3 - 5 * a * xx2 + 8 * a * xx - 4.0f * a,
                          0.0f));
}

Expr sinc(Expr x) {
    return sin(float(M_PI) * x) / x;
}

Expr kernel_lanczos(Expr x) {
    Expr value = sinc(x) * sinc(x/3);
    value = select(x == 0.0f, 1.0f, value); // Take care of singularity at zero
    value = select(x > 3 || x < -3, 0.0f, value); // Clamp to zero out of bounds
    return value;
}

struct KernelInfo {
    const char *name;
    float size;
    Expr (*kernel)(Expr);
};

static KernelInfo kernelInfo[] = {
    { "box", 0.5f, kernel_box },
    { "linear", 1.0f, kernel_linear },
    { "cubic", 2.0f, kernel_cubic },
    { "lanczos", 3.0f, kernel_lanczos }
};


InterpolationType interpolationType = LINEAR;
float scaleFactor = 1.0f;
int schedule = 3;

int main(int argc, char **argv) {
  Halide::Var x, y, c, k;
  Halide::ImageParam in(Halide::type_of<uint8_t>(), 3);
  Halide::ImageParam mean(Halide::type_of<float>(), 3);
  Halide::Param<int> output_width;
  Halide::Param<int> output_height;

  Func clamped_in = (BoundaryConditions::repeat_edge(in));
  Func flip;
  flip(x, y, c) = cast<float>(select(c == 0, clamped_in(x, y, 2),
                                     c == 1, clamped_in(x, y, 1),
                                     c == 2, clamped_in(x, y, 0),
                                     clamped_in(x, y, c)));

  //////////////////////////////////////////////////////////////////////////////
  /// Upscale
  float u_kernelSize = kernelInfo[interpolationType].size;
  Expr u_scaleFactorX = cast<float>(mean.width()) / in.width();
  Expr u_scaleFactorY = cast<float>(mean.height()) / in.height();

  // For downscaling, widen the interpolation kernel to perform lowpass
  // filtering.
  //Expr u_kernelScalingX = min(cast<float>(u_scaleFactorX), cast<float>(1.0f));
  Expr u_kernelScalingX = 1.0f;
  Expr u_kernelSizeX = u_kernelSize / u_kernelScalingX;

  //Expr u_kernelScalingY = min(cast<float>(u_scaleFactorY), cast<float>(1.0f));
  Expr u_kernelScalingY = 1.0f;
  Expr u_kernelSizeY = u_kernelSize / u_kernelScalingY;

  // source[xy] are the (non-integer) coordinates inside the source image
  Expr u_sourcex = (x + 0.5f) / u_scaleFactorX;
  Expr u_sourcey = (y + 0.5f) / u_scaleFactorY;

  // Initialize interpolation kernels. Since we allow an arbitrary
  // scaling factor, the filter coefficients are different for each x
  // and y coordinate.
  Func u_kernelx("kernelx"), u_kernely("kernely");
  Expr u_beginx = cast<int>(u_sourcex - u_kernelSizeX + 0.5f);
  Expr u_beginy = cast<int>(u_sourcey - u_kernelSizeY + 0.5f);
  RDom u_domx(0, cast<int>(2.0f * u_kernelSizeX) + 1, "domx");
  RDom u_domy(0, cast<int>(2.0f * u_kernelSizeY) + 1, "domy");
  {
    const KernelInfo &info = kernelInfo[interpolationType];
    Func kx, ky;
    kx(x, k) = info.kernel((k + u_beginx - u_sourcex) * u_kernelScalingX);
    ky(y, k) = info.kernel((k + u_beginy - u_sourcey) * u_kernelScalingY);
    u_kernelx(x, k) = kx(x, k) / sum(kx(x, u_domx));
    u_kernely(y, k) = ky(y, k) / sum(ky(y, u_domy));
  }

  // Perform separable resizing
  Func u_resized_x("resized_x");
  Func u_resized_y("resized_y");
  u_resized_x(x, y, c) = sum(u_kernelx(x, u_domx) *
                           (flip(u_domx + u_beginx, y, c)));
  u_resized_y(x, y, c) = sum(u_kernely(y, u_domy) *
                           u_resized_x(x, u_domy + u_beginy, c));

  /////////////////////////////////////////////////////////////////////////////
  /// Subtract mean
  Func clamped_mean = BoundaryConditions::repeat_edge(mean);
  Func clamped;
  clamped(x, y, c) =
    (clamp(u_resized_y(x, y, c), 0.0f, 1.0f) * 255.0f - clamped_mean(x, y, c))
    / 255.0f;

  /////////////////////////////////////////////////////////////////////////////
  /// Downscale
  float kernelSize = kernelInfo[interpolationType].size;

  Expr scaleFactorX = output_width / cast<float>(mean.width());
  Expr scaleFactorY = output_height / cast<float>(mean.height());

  // For downscaling, widen the interpolation kernel to perform lowpass
  // filtering.
  //Expr kernelScalingX = min(cast<float>(scaleFactorX), cast<float>(1.0f));
  Expr kernelScalingX = 1.0f;
  Expr kernelSizeX = kernelSize / kernelScalingX;

  //Expr kernelScalingY = min(cast<float>(scaleFactorY), cast<float>(1.0f));
  Expr kernelScalingY = 1.0f;
  Expr kernelSizeY = kernelSize / kernelScalingY;

  // source[xy] are the (non-integer) coordinates inside the source image
  Expr sourcex = (x + 0.5f) / scaleFactorX;
  Expr sourcey = (y + 0.5f) / scaleFactorY;

  // Initialize interpolation kernels. Since we allow an arbitrary
  // scaling factor, the filter coefficients are different for each x
  // and y coordinate.
  Func kernelx("kernelx"), kernely("kernely");
  Expr beginx = cast<int>(sourcex - kernelSizeX + 0.5f);
  Expr beginy = cast<int>(sourcey - kernelSizeY + 0.5f);
  RDom domx(0, cast<int>(2.0f * kernelSizeX) + 1, "domx");
  RDom domy(0, cast<int>(2.0f * kernelSizeY) + 1, "domy");
  {
    const KernelInfo &info = kernelInfo[interpolationType];
    Func kx, ky;
    kx(x, k) = info.kernel((k + beginx - sourcex) * kernelScalingX);
    ky(y, k) = info.kernel((k + beginy - sourcey) * kernelScalingY);
    kernelx(x, k) = kx(x, k) / sum(kx(x, domx));
    kernely(y, k) = ky(y, k) / sum(ky(y, domy));
  }

  // Perform separable resizing
  Func resized_x("resized_x");
  Func resized_y("resized_y");
  resized_x(x, y, c) = sum(kernelx(x, domx) *
                           cast<float>(clamped(domx + beginx, y, c)));
  resized_y(x, y, c) = sum(kernely(y, domy) *
                           resized_x(x, domy + beginy, c));

  Func final("final");
  final(x, y, c) = clamp(resized_y(x, y, c), 0.0f, 1.0f) * 255.0f;

  std::cout << "Finished function setup." << std::endl;

  // Scheduling
  bool parallelize = (schedule >= 2);
  bool vectorize = (schedule == 1 || schedule == 3);

  u_kernelx.compute_root();
  u_kernely.compute_at(clamped, y);

  kernelx.compute_root();
  kernely.compute_at(final, y);

  if (vectorize) {
    u_resized_x.vectorize(x, 4);
    clamped.compute_root();
    clamped.vectorize(x, 4);

    resized_x.vectorize(x, 4);
    final.vectorize(x, 4);
  }

  if (parallelize) {
    Var yo, yi;

    clamped.split(y, yo, y, 32).parallel(yo);
    u_resized_x.store_at(clamped, yo).compute_at(clamped, y);

    final.split(y, yo, y, 32).parallel(yo);
    resized_x.store_at(final, yo).compute_at(final, y);
  } else {
    u_resized_x.store_at(clamped, c).compute_at(clamped, y);
    resized_x.store_at(final, c).compute_at(final, y);
  }

  in.set_stride(0, 3);
  in.set_stride(2, 1);
  in.set_extent(2, 3);

  final.compile_to_file
    ("to_conv_patch",
     {in, mean,
         output_width, output_height});

  return 0;
}
