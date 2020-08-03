#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "argparse.h"
#include <chrono>

// #include "xtensor/xarray.hpp"
// #include "xtensor/xio.hpp"
// #include "xtensor/xadapt.hpp"

// https://stackoverflow.com/questions/62829519/converting-xtensor-xarray-to-opencv-mat-back-and-forth-in-cpp
// cv::Mat xarray_to_mat(xt::xarray<float> xarr)
// {
//     cv::Mat mat (xarr.shape()[0], xarr.shape()[1], CV_32FC1, xarr.data(), 0);
//     return mat;
// }

// xt::xarray<float> mat_to_xarray(cv::Mat mat)
// {
//     xt::xarray<float> res = xt::adapt(
//         (float*) mat.data, mat.cols * mat.rows, xt::no_ownership(), std::vector<std::size_t> {mat.rows, mat.cols});
//     return res;
// }

//def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
void letterbox(cv::Mat &img) {
  // default args
  int new_shape[] = {640, 640};
  auto color = cv::Scalar(114, 114, 114);
  const bool _auto=true;
  const bool scaleFill=false;
  const bool scaleup=true;

//    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
//    shape = img.shape[:2]  # current shape [height, width]
  int shape[] = {img.rows, img.cols};

    // if isinstance(new_shape, int):
    //     new_shape = (new_shape, new_shape)

    // # Scale ratio (new / old)
    auto r = std::min((float)new_shape[0] / shape[0], (float)new_shape[1] / shape[1]);
    // if not scaleup:  # only scale down, do not scale up (for better test mAP)
    if (!scaleup) {
        r = std::min(r, 1.0f);
    }

    // # Compute padding
    auto ratio = {r,r}; //# width, height ratios
    int new_unpad[] = {int(std::round(shape[1] * r)), int(std::round(shape[0] * r))};
    auto [dw,dh] = std::tuple<int,int>(new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]);
    if (_auto) {  //# minimum rectangle
        std::tie(dw, dh) = std::tuple<int,int>(dw % 64, dh % 64); // # wh padding
    } else if (scaleFill) {  //# stretch
        std::tie(dw, dh) = std::tuple<int,int>(0.0, 0.0);
        new_unpad[0] = new_shape[1];
        new_unpad[1] = new_shape[0];
        ratio = {(float)new_shape[1] / shape[1], (float)new_shape[0] / shape[0]};//  # width, height ratios
    }

    dw /= 2; //  # divide padding into 2 sides
    dh /= 2;

    // if shape[::-1] != new_unpad:  # resize
    if (shape[0] != new_unpad[1] || shape[1] != new_unpad[0]) {
        // img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR);
        cv::resize(img, img, cv::Size(new_unpad[0], new_unpad[1]), 0, 0, cv::INTER_LINEAR);
    }
    auto [top, bottom] = std::tuple<int,int>(int(std::round(dh - 0.1)), int(std::round(dh + 0.1)));
    auto [left, right] = std::tuple<int,int>(int(std::round(dw - 0.1)), int(std::round(dw + 0.1)));
    // img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    //return img; //, ratio, (dw, dh)
}

int main(int argc, const char* argv[]) {

  argparse::ArgumentParser parser("example", "Argument parser example");

  parser.add_argument().names({"--model"}).description("model weights").required(true);
  parser.add_argument().names({"--video"}).description("video to process").required(true);
  parser.enable_help();

  auto err = parser.parse(argc, argv);
  if (err) {
    std::cout << err << std::endl;
    return -1;
  }

  if (parser.exists("help")) {
    parser.print_help();
    return 0;
  }


  torch::jit::script::Module model;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(parser.get<std::string>("model"), torch::kCUDA);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  // model = model.cuda();

  std::cout << "ok\n";

  cv::VideoCapture capture(parser.get<std::string>("video"));
  cv::Mat frame;

  if( !capture.isOpened() )
      throw "Error when reading steam_avi";
  auto begin = std::chrono::steady_clock::now();
  for(int fidx=0; ;fidx++ )
  {
      capture >> frame;
      if(frame.empty())
          break;
      cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
      letterbox(frame);
      // std::cout << "frame.shape" << frame.rows << "x" << frame.cols << std::endl; // 448x640
      // break;
      // auto frame_nc = nc::NdArray<nc::uint8>(frame.data, frame.rows, frame.cols);
      // frame_nc = frame_nc.transpose();

      // xt::xarray<unsigned char> frame_xt = xt::adapt(
      //     (unsigned char*) frame.data, frame.cols * frame.rows, xt::no_ownership(), 
      //     std::vector<std::size_t> {frame.rows, frame.cols}
      //     );
      // xt::xarray<unsigned char> frame_xt1 = frame_xt.transpose({2,0,1});
      // frame_xt = xt::cast<float>(frame_xt) / 255

      auto image = torch::from_blob(frame.data, {frame.rows, frame.cols, 3}, torch::kU8);
      image = image.permute({2, 0, 1}); // HWC -> CHW
      image = image.to(torch::kFloat);
      image = image / 255.0;
      image = image.unsqueeze(0);
      image = image.cuda();


      // Create a vector of inputs.
      std::vector<torch::jit::IValue> inputs;
      // inputs.push_back(torch::ones({1, 3, 224, 224}));
      inputs.push_back(image);
      auto pred = model.forward(inputs).toTensorVector()[0];
      auto end = std::chrono::steady_clock::now();
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
      auto s = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
      std::cout << "fps= " << fidx << "/" << ms << "[ms] = " << float(fidx)/s << std::endl;

      // decltype(pred)::foo= 1;
      // std:: cout << "pred.shape " << pred.sizes() << std::endl;
      // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
      // TODO: apply NMS
  }

}

