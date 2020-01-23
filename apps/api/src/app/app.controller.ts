import { Controller, Get, Param, Res } from '@nestjs/common';

import { Message } from '@sedge/api-interfaces';

import { AppService } from './app.service';
import { join } from 'path';

@Controller()
export class AppController {
  constructor(private readonly appService: AppService) {}

  @Get('hello')
  getData(): Message {
    return this.appService.getData();
  }

  @Get('atlas/:imgId')
  test(@Param('imgId') imgId, @Res() res) {
    // console.log();
    // const imgPath = getImgPath(imgId);
    return res.sendFile(join(__dirname, 'assets', imgId));
    // return res
    // console.log(imgId);
  }
}
