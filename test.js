var addon = require('bindings')('people-detector');
var obj = new addon.Yolo_cpu();
// <arg1> = Input image file path with extension
// <arg2> = Output image file path with extension
// <arg3> = Size of first network of yolov3. The larger, the slower but the smaller objects are better detected.
// recommended: 320, 416, 608, 800 or other multiples of 32.
console.log( obj.start('image.jpeg', 'image-result.jpge', 416) );
