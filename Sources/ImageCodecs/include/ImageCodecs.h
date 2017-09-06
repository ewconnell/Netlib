//******************************************************************************
//  Created by Edward Connell on 10/1/16.
//  Copyright © 2016 Connell Research. All rights reserved.
//
#ifndef ImageCodecs_h
#define ImageCodecs_h

#include <stdlib.h>

//------------------------------------------------------------------------------
// This is used by the decoder to keep track of the source
// buffer position
typedef struct {
	// user buffer pointer, do not free!
	const unsigned char* data;
	size_t count;
	size_t pos;
} BufferPos;

typedef enum {
	CDataType_real8U,
	CDataType_real16U,
	CDataType_real16I,
	CDataType_real32I
} CDataType;

typedef enum {
	CStatus_success,
	CStatus_error
} CStatus;

typedef enum {
	CImageType_any,
	CImageType_gray,
	CImageType_grayAlpha,
	CImageType_rgb,
	CImageType_rgba
} CImageType;

typedef struct {
	void*         context;
	const size_t* dims;
	size_t        numDims;
	size_t        byteCount;
	CDataType     dataType;
	size_t        dataTypeSize;
	size_t        willChangeType;
	CStatus       status;
} CDecoderInfo;

typedef struct {
	const unsigned char* data;
	size_t count;
} CByteArray;

void freeCByteArray(CByteArray array);

//------------------------------------------------------------------------------
// png
CDecoderInfo pngDecodeInfo(const unsigned char* src, size_t count,
													 CImageType outImageType);
CStatus pngDecodeData(CDecoderInfo info, unsigned char* dst, size_t dstByteCount);
void freePngCDecoderInfo(CDecoderInfo info);

CByteArray pngEncode(const unsigned char* data, size_t elementCount,
										 const size_t* dims, size_t numDims, CDataType dataType,
										 long compression);

//------------------------------------------------------------------------------
// jpeg
CDecoderInfo jpegDecodeInfo(const unsigned char* src, size_t count,
														CImageType outImageType);
CStatus jpegDecodeData(CDecoderInfo info, unsigned char* dst, size_t dstByteCount);
void freeJpegCDecoderInfo(CDecoderInfo info);

CByteArray jpegEncode(const unsigned char* data, size_t elementCount,
											const size_t* dims, size_t numDims,	size_t quality);


#endif /* ImageCodecs_h */
