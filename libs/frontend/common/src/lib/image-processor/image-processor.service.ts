import { Injectable } from '@angular/core';
import * as Jimp from 'jimp';
import { from, Observable } from 'rxjs';
import { map as rxMap, flatMap as rxFlatMap } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class ImageProcessorService {
  constructor() {}
  public read(url: string): Observable<any> {
    return from(Jimp.read(url)).pipe(
      rxMap((image: any) => {
        image.getBase64AsObservable = (mime: string) => {
          return from(image.getBase64Async(mime));
        };

        image.getBufferAsObservable = (mime: string) => {
          return from(image.getBufferAsync(mime));
        };

        image.getImageDataAsObservable = (
          mime: string,
          width: number,
          height: number
        ) => {
          return from(image.getBase64AsObservable(mime)).pipe(
            rxFlatMap((base64: string) => {
              return new Observable(sub => {
                const canvas = document.createElement('canvas');
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                const img = new Image();
                img.onload = function() {
                  ctx.drawImage(img, 0, 0);
                  sub.next(ctx.getImageData(0, 0, width, height));
                  sub.complete();
                };
                img.src = base64;
              });
            })
          );
        };
        return image;
      })
    );
  }
}
