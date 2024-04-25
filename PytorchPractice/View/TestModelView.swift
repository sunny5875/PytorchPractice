//
//  TestModelView.swift
//  PytorchPractice
//
//  Created by 현수빈 on 4/23/24.
//
import SwiftUI

struct TestModelView: View {
    @State private var result: [String] = []
    
    var body: some View {
        NavigationView {
            Form {
                Section("test model 결과", content: {
                    ForEach(result, id: \.self) {
                        Text($0)
                    }
                })
            }
            .navigationTitle("Pytorch Practice")
        }
        .onAppear {
            Task {
                predict()
            }
        }
    }
    
    
    // MARK: - test model
    private var module2: TorchModule = {
        if let filePath = Bundle.main.path(forResource: "model", ofType: "ptl"),
           let module = TorchModule(fileAtPath: filePath) {
            return module
        } else {
            fatalError("Can't find the model file!")
        }
    }()
    
    private var sampleData: [String: Any] = {
        do {
            guard let filePath = Bundle.main.path(forResource: "result", ofType: "json")
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
        let x = sampleData["x"] as! [[Double]]
        let y = sampleData["mobile_y"] as! [[NSNumber]]
        
        for i in (0..<x.count) {
            guard
                let pixelBuffer = prepareBuffer(from: x[i]),
                let outputs = module2.predict(buffer: pixelBuffer)
            else { return }
            
            let floatResult = convertNSMutableArrayToFloatArray(outputs)
            let answer = y[i].map {Int(truncating: $0)}
            
            var compareResult: Bool = true
            for j in (0..<128) {
                if roundToBinary(floatResult[j]) != answer[j] {
                    compareResult = false
                    break
                }
            }
            result.append("\(i+1)th 결과: \(compareResult)")
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
    
    //    func convertToRawPointer(_ array: [Double]) -> UnsafeMutableRawPointer? {
    //        // Calculate the total size of the buffer needed in bytes
    //        let bufferSize = array.count * MemoryLayout<Double>.stride
    //
    //        // Allocate memory for the buffer using UnsafeMutableRawPointer
    //        let buffer = UnsafeMutableRawPointer.allocate(byteCount: bufferSize, alignment: MemoryLayout<Double>.alignment)
    //
    //        // Bind the memory to the Double type
    //        let typedBuffer = buffer.bindMemory(to: Double.self, capacity: array.count)
    //
    //        // Copy the elements of the array into the buffer
    //        array.withUnsafeBytes { (arrayBuffer: UnsafeRawBufferPointer) in
    //            guard let sourcePointer = arrayBuffer.baseAddress else {
    //                return
    //            }
    //            typedBuffer.initialize(from: sourcePointer.bindMemory(to: Double.self, capacity: array.count), count: array.count)
    //        }
    //
    //        // Return the UnsafeMutableRawPointer to the allocated and initialized buffer
    //        return buffer
    //    }
    
    private func convertNSMutableArrayToFloatArray(_ nsArray: NSMutableArray) -> [Float] {
        var floatArray = [Float]()
        
        for case let number as NSNumber in nsArray {
            floatArray.append(number.floatValue)
        }
        
        return floatArray
    }
    
}
