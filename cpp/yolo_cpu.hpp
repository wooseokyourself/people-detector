#ifndef YOLO_CPU
#define YOLO_CPU

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#include <opencv2/core/mat.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

int postProcess (Mat& frame, const vector<Mat>& outs, Net& net, float confThreshold, float nmsThreshold);

void imagePadding (Mat& frame);

int doInference (string INPUT_IMAGE_PATH, int resize);

#endif