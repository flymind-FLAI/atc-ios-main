import Foundation

struct Model: Identifiable {
    var id = UUID()
    var name: String
    var info: String
    var url: String
    var filename: String
    var bundleResource: String?

    var fileURL: URL {
        if let bundleResource,
           let bundleURL = Bundle.main.url(forResource: bundleResource, withExtension: "bin", subdirectory: "models") {
            return bundleURL
        }
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(filename)
    }

    func fileExists() -> Bool {
        FileManager.default.fileExists(atPath: fileURL.path)
    }
}
