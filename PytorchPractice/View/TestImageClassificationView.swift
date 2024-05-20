//
//  TestImageClassificationView.swift
//  PytorchPractice
//
//  Created by 현수빈 on 4/18/24.
//
import SwiftUI


struct TestImageClassificationView: View {
    @State var result: String = "image classification"
    
    @State var image = UIImage(named: "image")
    
    var body: some View {
        NavigationView {
            Form {
                VStack {
                    Image("image")
                        .imageScale(.large)
                        .foregroundStyle(.tint)
                    Text(result)
                }
            }
            .navigationTitle("Pytorch Practice")
        }
        .onAppear {
            predict()
        }
    }
    
    // MARK: - image classification
    private var module: TorchModule  {
        if let filePath = Bundle.main.path(forResource: "model 1", ofType: "ptl"),
           let module = TorchModule(fileAtPath: filePath) {
            return module
        } else {
            fatalError("Can't find the model file!")
        }
    }
    
    private var labels: [String] {
        if let filePath = Bundle.main.path(forResource: "words", ofType: "txt"),
           let labels = try? String(contentsOfFile: filePath) {
            return labels.components(separatedBy: .newlines)
        } else {
            fatalError("Can't find the text file!")
        }
    }
    
    func predict() {
        let resizedImage = image!.resized(to: CGSize(width: 224, height: 224))
        guard var pixelBuffer = resizedImage.normalized() else {
            return
        }
        guard let outputs = module.predict(image: UnsafeMutableRawPointer(&pixelBuffer)) else {
            return
        }
        let zippedResults = zip(labels.indices, outputs)
        let sortedResults = zippedResults.sorted { $0.1.floatValue > $1.1.floatValue }.prefix(3)
        var text = ""
        for result in sortedResults {
            text += "\u{2022} \(labels[result.0]) \n\n"
        }
        result = text
    }
}
