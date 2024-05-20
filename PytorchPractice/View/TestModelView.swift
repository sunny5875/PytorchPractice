//
//  TestModelView.swift
//  PytorchPractice
//
//  Created by 현수빈 on 4/23/24.
//
import SwiftUI
import PhotosUI

struct TestModelView: View {
    
    private let modelList: [Int] = [1, 2, 3, 4, 5 ]
    @State private var result: [String] = []
    
    @State private var avatarItem: PhotosPickerItem?
    @State private var avatarImage: Image?
    
    
    @State var isShowing = false
    @State private var selectedModelIndex: Int = 0
    
    var body: some View {
        NavigationView {
            Form {
                Section("test model 선택", content: {
                    ForEach(modelList, id: \.self) { item in
                        Text("model \(item)")
                            .onTapGesture {
                                selectedModelIndex = item
                                Task {
                                    predict()
                                }
                            }
                    }
                })
                
                Section("test model \(selectedModelIndex)결과", content: {
                    ForEach(result, id: \.self) {
                        Text($0)
                    }
                })
            }
            .navigationTitle("Pytorch Practice")
            
        }
    }
    
    
    // MARK: - test model
    
    private var testData: [String: Any] = {
        do {
            guard let filePath = Bundle.main.path(forResource: "verify", ofType: "json")
            else { return [:] }
            let data = try Data(contentsOf: URL(fileURLWithPath: filePath), options: .mappedIfSafe)
            
            guard let jsonDictionary = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] else {
                print("Failed to parse JSON data.")
                return [:]
            }
            
            return jsonDictionary
        } catch {
            print("Error reading JSON file: \(error)")
            return [:]
        }
    }()
    
    private var answerInputData: [String: Any] = {
        do {
            guard let filePath = Bundle.main.path(forResource: "train", ofType: "json")
            else { return [:] }
            let data = try Data(contentsOf: URL(fileURLWithPath: filePath), options: .mappedIfSafe)
            
            guard let jsonDictionary = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] else {
                print("Failed to parse JSON data.")
                return [:]
            }
            
            return jsonDictionary
        } catch {
            print("Error reading JSON file: \(error)")
            return [:]
        }
    }()
    
    
    
    
    private func predict() {
        // model 가져오기
        guard let filePath = Bundle.main.path(forResource: "model \(selectedModelIndex)", ofType: "ptl"),
              let module = TorchModule(fileAtPath: filePath) else { return }
        
        guard let answerInput = answerInputData["sample \(selectedModelIndex)"] as? [Double],
            let pixelBuffer = prepareBuffer(from: answerInput),
            let outputs = module.predict(buffer: pixelBuffer)
        else { return }
        
        
        let answer = convertNSMutableArrayToFloatArray(outputs).map {roundToBinary($0)}
        
        for i in 1..<6 {
            guard let x = testData["sample \(i)"] as? [[Double]] else { return }
            
            for j in (0..<x.count) {
                guard
                    let pixelBuffer = prepareBuffer(from: x[j]),
                    let outputs = module.predict(buffer: pixelBuffer)
                else { return }
                
                let floatResult = convertNSMutableArrayToFloatArray(outputs)
                
                let tempResult = floatResult.map {roundToBinary($0)}
                
                result.append("image \(i)-\(j+1)로 검증: \(tempResult == answer)")
                
            }
        }
    }
    
}

extension TestModelView {
    
    private func roundToBinary(_ value: Float) -> Int {
        let threshold: Float = 0.5
        return value >= threshold ? 1 : 0
    }
    
    private func prepareBuffer(from doubleArray: [Double]) -> UnsafeMutableRawPointer? {
        let bufferSize = doubleArray.count * MemoryLayout<Float>.stride
        
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: bufferSize, alignment: MemoryLayout<Float>.alignment)
        
        let typedBuffer = buffer.bindMemory(to: Float.self, capacity: doubleArray.count)
        
        for (index, value) in doubleArray.enumerated() {
            typedBuffer[index] = Float(value)
        }
        
        return buffer
    }
    
    private func convertNSMutableArrayToFloatArray(_ nsArray: NSMutableArray) -> [Float] {
        var floatArray = [Float]()
        
        for case let number as NSNumber in nsArray {
            floatArray.append(number.floatValue)
        }
        
        return floatArray
    }
}
