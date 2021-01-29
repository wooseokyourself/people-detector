# people-detector
Module for node.js, based on OpenCV DNN, YOLOv3 and COCO dataset.

# Dependencies
- Linux
- OpenCV 4 (with pkgconfig)
- node-gyp

# Install
1. Initialize project
~~~
$ npm init -y
~~~


2. Install node packages in project
~~~
$ npm install bindings node-addon-api
~~~


3. Configure node-gyp
~~~
$ node-gyp configure
~~~


4. Build module
~~~
$ node-gyp build
~~~


5. Get YOLOv3 pre-trained model
~~~
$ chmod 755 getModel.sh
$ sh getModel.sh
~~~

# Usage
see ```test.js```. It returns the number of people in image and write result image.

# Others
+ If you want to use model of yolov4, then OpenCV 4.4 or later version is required.
