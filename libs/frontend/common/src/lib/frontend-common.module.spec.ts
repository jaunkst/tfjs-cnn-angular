import { async, TestBed } from '@angular/core/testing';
import { FrontendCommonModule } from './frontend-common.module';

describe('FrontendCommonModule', () => {
  beforeEach(async(() => {
    TestBed.configureTestingModule({
      imports: [FrontendCommonModule]
    }).compileComponents();
  }));

  it('should create', () => {
    expect(FrontendCommonModule).toBeDefined();
  });
});
