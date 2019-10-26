//******************************************************************************
// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
