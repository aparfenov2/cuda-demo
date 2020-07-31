#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

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
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-model>\n";
    return -1;
  }


  torch::jit::script::Module model;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  // model = model.cuda();

  std::cout << "ok\n";

  std::string filename = "yourfile.avi";
  cv::VideoCapture capture(filename);
  cv::Mat frame;

  if( !capture.isOpened() )
      throw "Error when reading steam_avi";

  for( ; ; )
  {
      capture >> frame;
      if(frame.empty())
          break;
      cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
      letterbox(frame);

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
      auto pred = model.forward(inputs).toTensor();
      std:: cout << "pred.shape " << pred.dim() << std::endl;
      // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
      // TODO: apply NMS
  }

}

