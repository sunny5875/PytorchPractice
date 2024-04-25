//
//  MainTabView.swift
//  PytorchPractice
//
//  Created by 현수빈 on 4/23/24.
//

import SwiftUI

struct MainTabView: View {
    var body: some View {
        TabView(selection: .constant(2)) {
            TestImageClassificationView()
                .tabItem {
                    VStack {
                        Image(systemName: "1.square.fill")
                        Text("Image classification")
                    }
                }.tag(1)
            TestModelView()
                .tabItem {
                    VStack {
                        Image(systemName: "2.square.fill")
                        Text("Test model")
                    }
                }.tag(2)
        }
    }
}
