//******************************************************************************
//  Created by Edward Connell on 9/21/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import Zlib

#if os(Linux)
import Glibc
#endif


//============================================================================
// extract
public func extract(zip: URL, to destURL: URL, totalItems: inout Int,
                    progress handler: (Int) -> Void) throws
{
	//Open the ZIP archive
	var err: Int32 = 0
	let za = zip_open(zip.path, 0, &err)
	guard err == 0 else {
		throw ModelError.error(getZipErrorString(error: err))
	}
	defer { zip_close(za) }
	totalItems = Int(zip_get_num_entries(za, 0))

	// loop through the entries
	var stat = zip_stat_t()
	for i in 0..<totalItems {
		handler(i)
		if zip_stat_index(za, UInt64(i), 0, &stat) == 0 {
			let name = String(cString: stat.name)
			let itemSize = Int(stat.size)
			let itemURL = destURL.appendingPathComponent(name)

			// check for directory
			if name.characters.last == "/" {
				try FileManager.default.createDirectory(
					atPath: itemURL.path, withIntermediateDirectories: true, attributes: nil)

			} else {
				let zf = zip_fopen_index(za, UInt64(i), 0)
				defer { zip_fclose(zf) }
				if zf == nil { throw ModelError.error("Failed to open zip element: \(name)") }

				// read archive data
				var data = Data(count: itemSize)
				_ = data.withUnsafeMutableBytes {
					zip_fread(zf, UnsafeMutableRawPointer($0), stat.size)
				}

				// write to output file
				//  Note: Data.write(to: is insanely slow. Something is wrong with it
				//        at this time on Linux
				let fd = open(itemURL.path, O_RDWR | O_TRUNC | O_CREAT, 0o664)
				defer { close(fd) }
				if fd < 0 { throw ModelError.error("failed to open: \(itemURL.path)") }

				if (data.withUnsafeBytes { write(fd, $0, data.count) }) != data.count {
					throw ModelError.error("failed to write: \(itemURL.path)")
				}
			}
		}
	}
}

private func getZipErrorString(error: Int32) -> String {
	var buffer = [UInt8](repeating: 0, count: 256)
	_ = buffer.withUnsafeMutableBufferPointer { bp in
		bp.baseAddress!.withMemoryRebound(to: Int8.self, capacity: bp.count) {
			zip_error_to_str($0, zip_uint64_t(bp.count), error, errno)
		}
	}
	return String(bytes: buffer, encoding: .utf8)!
}

//============================================================================
// unzip
public func unzip(data: [UInt8]) throws -> [UInt8]
{
	return try unzip(buffer: data.withUnsafeBufferPointer {$0})
}

//============================================================================
// unzip
public func unzip(buffer: BufferUInt8) throws -> [UInt8]
{
	guard buffer.count > 0 else { return [UInt8]() }
	var stream = createZStream(buffer: buffer)
	var status: Int32
	
	status = inflateInit2_(&stream, MAX_WBITS + Int32(32), ZLIB_VERSION,
	                       Int32(MemoryLayout<z_stream>.size))
	
	guard status == Z_OK else {
		// inflateInit2 returns:
		// Z_VERSION_ERROR   The zlib library version is incompatible
		//                   with the version assumed by the caller.
		// Z_MEM_ERROR       There was not enough memory.
		// Z_STREAM_ERROR    A parameters are invalid.
		throw GzipError(code: status, msg: stream.msg)
	}
	
	var result = [UInt8](repeating: 0, count: buffer.count * 2)
	
	while status == Z_OK {
		// grow buffer if needed
		if Int(stream.total_out) >= result.count {
			result.append(contentsOf: repeatElement(UInt8(0), count: buffer.count / 2))
		}

		// update remaining available buffer space
		stream.avail_out = uInt(result.count) - uInt(stream.total_out)

		// point to next available buffer space
		result.withUnsafeMutableBufferPointer {
			let ptr = $0.baseAddress!.advanced(by: Int(stream.total_out))
			ptr.withMemoryRebound(to: Bytef.self, capacity: Int(stream.avail_out)) {
				stream.next_out = $0
			}
		}

		// inflate the next portion
		status = inflate(&stream, Z_SYNC_FLUSH)
	}
	
	guard inflateEnd(&stream) == Z_OK && status == Z_STREAM_END else {
		// inflate returns:
		// Z_DATA_ERROR   The input was corrupted (input stream not conforming
		//                to the zlib format or incorrect check value).
		// Z_STREAM_ERROR The stream structure was inconsistent
		//                (for example if next_in or next_out was NULL).
		// Z_MEM_ERROR    There was not enough memory.
		// Z_BUF_ERROR    No progress is possible or there was not enough room
		//                in the output buffer when Z_FINISH is used.
		throw GzipError(code: status, msg: stream.msg)
	}
	
	// remove unused buffer space
	result.removeLast(result.count - Int(stream.total_out))
	return result
}

//------------------------------------------------------------------------------
// createZStream
private func createZStream(buffer: BufferUInt8) -> z_stream
{
	let ptr = buffer.baseAddress!
		.withMemoryRebound(to: Bytef.self, capacity: buffer.count){$0}

	return z_stream(
		next_in: UnsafeMutablePointer(mutating: ptr),
		avail_in: uint(buffer.count),
		total_in: 0,
		next_out: nil,
		avail_out: 0,
		total_out: 0,
		msg: nil,
		state: nil,
		zalloc: nil,
		zfree: nil,
		opaque: nil,
		data_type: 0,
		adler: 0,
		reserved: 0
	)
}

//============================================================================
// zip
public func zip(data: [UInt8], compression: Int = -1) throws -> [UInt8] {
	return try zip(buffer: data.withUnsafeBufferPointer {$0}, compression: compression)
}

//============================================================================
// zip
public func zip(buffer: BufferUInt8, compression: Int = -1) throws -> [UInt8] {
	guard buffer.count > 0 else { return [UInt8]() }
	var stream = createZStream(buffer: buffer)
	var status: Int32
	
	// set compression
	let level = (0...9 ~= compression) ? Int32(compression) : Z_DEFAULT_COMPRESSION
	
	status = deflateInit2_(&stream, level, Z_DEFLATED,
	                       MAX_WBITS + Int32(16), MAX_MEM_LEVEL, Z_DEFAULT_STRATEGY,
	                       ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size))

	guard status == Z_OK else {
		// deflateInit2 returns:
		// Z_VERSION_ERROR  The zlib library version is incompatible with
		//                  the version assumed by the caller.
		// Z_MEM_ERROR      There was not enough memory.
		// Z_STREAM_ERROR   A parameter is invalid.
		throw GzipError(code: status, msg: stream.msg)
	}
	
	var result = [UInt8](repeating: 0, count: 16.KB)

	while stream.avail_out == 0 {
		if Int(stream.total_out) >= result.count {
			result.append(contentsOf: repeatElement(UInt8(0), count: 16.KB))
		}
		
		// update remaining available buffer space
		stream.avail_out = uInt(result.count) - uInt(stream.total_out)
		
		// point to next available buffer space
		result.withUnsafeMutableBufferPointer {
			let ptr = $0.baseAddress!.advanced(by: Int(stream.total_out))
			ptr.withMemoryRebound(to: Bytef.self, capacity: Int(stream.avail_out)) {
				stream.next_out = $0
			}
		}
		
		// deflate the next portion
		deflate(&stream, Z_FINISH)
	}

	deflateEnd(&stream)
	
	// remove unused buffer space
	result.removeLast(result.count - Int(stream.total_out))
	return result
}

//============================================================================
// GzipError
// http://www.zlib.net/manual.html
//
public enum GzipError: Error
{
	case stream(message: String)
	case data(message: String)
	case memory(message: String)
	case buffer(message: String)
	case version(message: String)
	case unknown(message: String, code: Int)
	
	public init(code: Int32, msg: UnsafePointer<CChar>)
	{
		let message =  String(cString: msg)
		
		switch code {
		case Z_STREAM_ERROR:
			self = .stream(message: message)
			
		case Z_DATA_ERROR:
			self = .data(message: message)
			
		case Z_MEM_ERROR:
			self = .memory(message: message)
			
		case Z_BUF_ERROR:
			self = .buffer(message: message)
			
		case Z_VERSION_ERROR:
			self = .version(message: message)
			
		default:
			self = .unknown(message: message, code: Int(code))
		}
	}
}










