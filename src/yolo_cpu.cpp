#include "yolo_cpu.hpp"

Napi::Object Yolo_cpu::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func =
        DefineClass(env,
                    "Yolo_cpu",
                    {InstanceMethod("start", &Yolo_cpu::start)});

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("Yolo_cpu", func);
    return exports;
}

Yolo_cpu::Yolo_cpu(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Yolo_cpu>(info) {
    this->MODEL_PATH = "bin/model/yolov3.weights";
    this->CFG_PATH = "bin/model/yolov3.cfg";
    this->CLASSES_PATH = "bin/model/coco.names";

    this->confThreshold = 0.4;
    this->nmsThreshold = 0.5;

    this->net = readNet(MODEL_PATH, CFG_PATH);
    this->net.setPreferableBackend(DNN_BACKEND_OPENCV);
    this->net.setPreferableTarget(DNN_TARGET_CPU);
    this->outNames = net.getUnconnectedOutLayersNames();
}
Napi::Value Yolo_cpu::start(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Wrong number of arguments").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (!info[0].IsString() || !info[1].IsString() || !info[2].IsNumber()) {
        Napi::TypeError::New(env, "Wrong arguments").ThrowAsJavaScriptException();
        return env.Null();
    }
    string arg0 = info[0].As<Napi::String>().Utf8Value();
    string arg1 = info[1].As<Napi::String>().Utf8Value();
    int arg2 = info[2].As<Napi::Number>().Int32Value();
    string arg3 = " ";
    if (info.Length() == 4)
        arg3 = info[3].As<Napi::String>().Utf8Value();
    
    int result = this->doInference(arg0, arg1, arg2, arg3);

    Napi::Number ret = Napi::Number::New(env, result);
    return ret;
}

int Yolo_cpu::doInference(const string inputImagePath, const string outputImagePath, const int resize, const string roiInfo) {
    Mat frame = imread(inputImagePath, IMREAD_COLOR); 
    vector<Mat> outs;
    int camID = inputImagePath[inputImagePath.size()-6]; // "...$(ID).jpeg"

//Mark: Pre-process
    preProcess(frame, camID, resize, roiInfo);

//Mark: Go inference
    net.forward(outs, outNames);

//Mark: Post-process
    int peopleNum = postProcess(frame, outs);

//Mark: Draw rect and other info in output image.
    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    
    string labelInferTime = format ("Inference time: %.2f ms", t);
    string labelPeople = format ("People: %d", peopleNum);
    putText (frame, labelInferTime, Point(0, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
    putText (frame, labelPeople, Point(0, 70), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);

    imwrite(outputImagePath, frame);
    return peopleNum;
}

void Yolo_cpu::preProcess(Mat& frame, const int& camID, const int& resize, const string& roiInfo) {
    /**
     * roiInfo에 담긴 데이터는 frame 이 640X480 일 때를 기준으로 생성된 것이므로,
     * 이 데이터를 현재 frame의 해상도에 맞게 변환해주어야함
    */
    //Mark: remove roi image
    int originW = frame.cols;
    int originH = frame.rows;

    Json::Reader reader;
    Json::Value root;
    reader.parse(roiInfo, root);
    Json::Value infoArray = root["data"];
    for (int i=0; i<infoArray.size(); i++) {
        if (infoArray[i]["id"].asInt() == camID) {
            // 640X480 기준 좌표
            int leftTopX = infoArray[i]["leftX"].asInt();
            int leftTopY = infoArray[i]["leftY"].asInt();
            int rightBottomX = leftTopX + infoArray[i]["width"].asInt();
            int rightBottomY = leftTopY + infoArray[i]["height"].asInt();

            // 원본 해상도 기준 좌표
            int originLeftTopX = (leftTopX * originW) / 640;
            int originLeftTopY = (leftTopY * originH) / 480;
            int originRightBottomX = (rightBottomX * originW) / 640;
            int originRightBottomY = (rightBottomY * originW) / 480;
            
            rectangle(frame, Point(originLeftTopX, originLeftTopY), Point(originRightBottomX, originRightBottomY), Scalar(255, 255, 255), FILLED);
        }
    }

    //Mark: Image padding
    if (frame.rows == frame.cols)
        return;

    int length = frame.cols > frame.rows ? frame.cols : frame.rows;
    if (frame.cols < length) {
        Mat pad(length, length - frame.cols, frame.type(), Scalar(255, 255, 255));
        hconcat(pad, frame, frame);
    }
    else {
        Mat pad(length - frame.rows, length, frame.type(), Scalar(255, 255, 255));
        vconcat(pad, frame, frame);
    }

    //Mark: Prepare for inference
    static Mat blob = blobFromImage(frame, 
                                    1, // scalarfactor: double
                                    Size(resize, resize), // resizeRes: Size
                                    Scalar(), 
                                    true, // swapRB: bool
                                    false, 
                                    CV_8U);

    net.setInput(blob,
                 "", 
                 1/255.0, // scale: double
                 Scalar()); // mean: Scalar
}

int Yolo_cpu::postProcess(Mat& frame, const vector<Mat>& outs) {
    int people = 0;
    static vector<int> outLayers = net.getUnconnectedOutLayers();
    static string outLayerType = net.getLayer(outLayers[0])->type;

    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    if (outLayerType == "DetectionOutput") {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() > 0);
        for (size_t k = 0; k < outs.size(); k++) {
            float* data = (float*)outs[k].data;
            for (size_t i = 0; i < outs[k].total(); i += 7) {
                float confidence = data[i + 2];
                if (confidence > confThreshold) {
                    int left   = (int)data[i + 3];
                    int top    = (int)data[i + 4];
                    int right  = (int)data[i + 5];
                    int bottom = (int)data[i + 6];
                    int width  = right - left + 1;
                    int height = bottom - top + 1;
                    if (width <= 2 || height <= 2) {
                        left   = (int)(data[i + 3] * frame.cols);
                        top    = (int)(data[i + 4] * frame.rows);
                        right  = (int)(data[i + 5] * frame.cols);
                        bottom = (int)(data[i + 6] * frame.rows);
                        width  = right - left + 1;
                        height = bottom - top + 1;
                    }
                    classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                    boxes.push_back(Rect(left, top, width, height));
                    confidences.push_back(confidence);
                }
            }
        }
    }
    else if (outLayerType == "Region") {
        for (size_t i = 0; i < outs.size(); ++i) {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {   
        int idx = indices[i];
        if (classIds[idx] == 0) { // Draw rectangle if class is a person.
            people++;
            Rect box = boxes[idx];
            int left = box.x;
            int top = box.y;
            int right = box.x + box.width;
            int bottom = box.y + box.height;
            rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));
        }
    }
    return people;
}