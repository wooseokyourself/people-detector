var addon = require('bindings')('people-detector');
var obj = new addon.Yolo_cpu();
console.log( obj.start('image.jpeg', 'image-result.jpeg', 416) ); // people number 
