import {
  Component,
  OnInit,
  AfterViewInit,
  ViewChild,
  ElementRef
} from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import * as Jimp from 'jimp';

import { from, of, combineLatest } from 'rxjs';
import {
  map as rxMap,
  tap as rxTap,
  startWith as rxStartWith,
  take as rxTake
} from 'rxjs/operators';
import { prop } from 'ramda';
import { FormGroup, FormControl } from '@angular/forms';

const TARGET_SHAPE_WIDTH = 250;
const TARGET_SHAPE_HEIGHT = 250;
const TARGET_SHAPE_CHANNELS = 3;
const TARGET_SHAPE = [
  TARGET_SHAPE_WIDTH,
  TARGET_SHAPE_HEIGHT,
  TARGET_SHAPE_CHANNELS
];
const NUM_OUTPUT_CLASSES = 3;

@Component({
  selector: 'sedge-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements AfterViewInit {
  @ViewChild('canvas', { static: true })
  public canvas: ElementRef<HTMLCanvasElement>;

  public inputForm = new FormGroup({
    inputValue: new FormControl(0)
  });

  public prediction$ = combineLatest([
    this.getModelObs().pipe(rxMap(model => {})),
    this.inputForm.valueChanges.pipe(rxMap(prop('inputValue')))
  ]).pipe(
    rxMap(([model, inputValue]: [any, number]) => {
      return inputValue;
    })
  );

  // this.inputForm.valueChanges.pipe(
  //   rxMap(prop('inputValue')),
  //   rxMap((inputValue: number) => {
  //     const output = this.linearModel.predict(
  //       tf.tensor2d([inputValue], [1, 1])
  //     ) as any;
  //     return Array.from(output.dataSync())[0];
  //   })
  // );

  constructor() {}

  ngAfterViewInit() {
    // this.photoAPI$.subscribe((photon: any) => {
    // console.log(window['Jimp']);

    // Jimp.read('https://i.picsum.photos/id/52/200/300.jpg', (err, lenna) => {
    //   if (err) throw err;
    //   lenna
    //     .resize(256, 256) // resize
    //     .quality(60) // set JPEG quality
    //     .greyscale() // set greyscale
    //     .getBase64(); // save
    // });

    Jimp.read('https://i.picsum.photos/id/52/200/300.jpg')
      .then(image => {
        const ctx = this.canvas.nativeElement.getContext('2d');
        image
          .resize(64, 64)
          .grayscale()
          .getBase64Async('image/jpeg')
          .then(base64 => {
            // console.log(base64);
            const img = new Image();
            img.onload = function() {
              console.log('onload');
              ctx.drawImage(img, 0, 0);
            };
            img.src = base64;
            // image.l
            // console.log({ buffer });
            // ctx.drawImage(img, 50, 50);
            // image.src =
            //   'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAAADhCAMAAAAJbSJIAAAAA1BMVEXtGySTdVFiAAAASElEQVR4nO3BgQAAAADDoPlTX+AIVQEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADwDcaiAAFXD1ujAAAAAElFTkSuQmCC';
          });

        // ctx.drawImage(image, 0, 0);
        // ctx.beginPath();
        // ctx.rect(20, 20, 150, 100);
        // ctx.fillStyle = 'red';
        // ctx.fill();

        // Do stuff with the image.
      })
      .catch(err => {
        // Handle an exception.
      });

    // Create a canvas and get a 2D context from the canvas
    // var canvas = document.getElementById('canvas') as any;

    // Module has now been imported.
    // All image processing logic w/ Photon goes here.
    // See sample code below.
    // });
  }

  public getModelObs() {
    const model = tf.sequential();

    model.add(
      tf.layers.conv2d({
        inputShape: TARGET_SHAPE,
        kernelSize: 3,
        filters: 16,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
      })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    model.add(tf.layers.flatten());

    model.add(
      tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
      })
    );

    return of(model);
  }

  // public train() {
  //   this.linearModel.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  //   this.linearModel.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

  //   // Training data, completely random stuff
  //   const xs = tf.tensor1d([3.2, 4.4, 5.5]);
  //   const ys = tf.tensor1d([1.6, 2.7, 3.5]);

  //   console.log('model trained!');

  //   // Train
  //   return from(this.linearModel.fit(xs, ys));
  // }
}
