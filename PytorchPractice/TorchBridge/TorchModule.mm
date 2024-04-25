#import "TorchModule.h"
#import <Libtorch-Lite-Nightly/Libtorch-Lite.h>
#include <caffe2/core/timer.h>

@implementation TorchModule {
@protected
    torch::jit::mobile::Module _impl;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
    self = [super init];
    if (self) {
        try {
            _impl = torch::jit::_load_for_mobile(filePath.UTF8String);
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}

- (NSArray<NSNumber*>*)predictImage:(void*)imageBuffer {
    try {
        at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, 224, 224}, at::kFloat);
        c10::InferenceMode guard;
        caffe2::Timer t;
        auto outputTensor = _impl.forward({tensor}).toTensor().cpu();
        std::cout << "forward took: " << t.MilliSeconds() << std::endl;
        float* floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) {
            return nil;
        }
        NSMutableArray* results = [[NSMutableArray alloc] init];
        for (int i = 0; i < 1000; i++) {
            [results addObject:@(floatBuffer[i])];
        }
        return [results copy];
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

- (NSMutableArray*)predict:(void*) buffer {
    try {
        at::Tensor tensor = torch::from_blob(buffer, {1, 1, 1, 128}, at::kFloat);
        //        std::cout << tensor << std::endl;
        
        if (!tensor.is_contiguous()) {
            NSLog(@"Input tensor is not contiguous.");
            return nil;
        }
        
        c10::InferenceMode guard;
        caffe2::Timer t;
        auto outputTensor = _impl.forward({tensor}).toTensor().cpu();
        //        std::cout << outputTensor << std::endl;
        
        if (outputTensor.numel() != 128) {
            NSLog(@"Output tensor size mismatch.");
            return nil;
        }
        std::cout << "소요 시간: " << t.MilliSeconds() << "ms" << std::endl;
        float* floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) {
            return nil;
        }
        NSMutableArray* results = [[NSMutableArray alloc] init];
        for (int i = 0; i < 128; i++) {
            [results addObject:@(floatBuffer[i])];
        }
        return [results copy];
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

@end
