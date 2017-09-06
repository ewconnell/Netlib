//******************************************************************************
//  Created by Edward Connell on 6/2/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import ImageCodecs

final public class ImageFormat : ModelObjectBase, InitHelper {
	//----------------------------------------------------------------------------
	// properties
	public var channelFormat = ChannelFormat.any	{ didSet{onSet("channelFormat")}}
	public var compression = -1                   { didSet{onSet("compression")} }
	public var dataType = DataType.real8U	        { didSet{onSet("dataType")} }
	public var encoding = ImageEncoding.any	      { didSet{onSet("encoding")} }
	
	// maybe move options into separate structure?
	public var jpegQuality = -1                   { didSet{onSet("jpegQuality")} }

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "channelFormat",
		            get: { [unowned self] in self.channelFormat },
		            set: { [unowned self] in self.channelFormat = $0 })
		addAccessor(name: "compression",
		            get: { [unowned self] in self.compression },
		            set: { [unowned self] in self.compression = $0 })
		addAccessor(name: "dataType",
		            get: { [unowned self] in self.dataType },
		            set: { [unowned self] in self.dataType = $0 })
		addAccessor(name: "encoding",
		            get: { [unowned self] in self.encoding },
		            set: { [unowned self] in self.encoding = $0 })
		addAccessor(name: "jpegQuality",
		            get: { [unowned self] in self.jpegQuality },
		            set: { [unowned self] in self.jpegQuality = $0 })
	}
}

//------------------------------------------------------------------------------
// ChannelFormat
extension ChannelFormat {
	public init(type: CImageType) {
		switch type {
		case CImageType_any      : self = .any
		case CImageType_gray     : self = .gray
		case CImageType_grayAlpha: self = .grayAlpha
		case CImageType_rgb      : self = .rgb
		case CImageType_rgba     : self = .rgba
		default: fatalError()
		}
	}

	public var ctype: CImageType {
		switch self {
		case .any      : return CImageType_any
		case .gray     : return CImageType_gray
		case .grayAlpha: return CImageType_grayAlpha
		case .rgb      : return CImageType_rgb
		case .rgba     : return CImageType_rgba
		}
	}
}

public enum ImageEncoding : String, EnumerableType {
	case any, jpeg, png
}
