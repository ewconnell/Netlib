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
// See http://www.libpng.org/pub/png/libpng-manual.txt
//     http://www.libpng.org/pub/png/spec/1.1/PNG-ColorAppendix.html
//
#include "include/ImageCodecs.h"
#include <png.h>
#include <assert.h>
#include <memory.h>
#include <zlib.h>

// axes
#define rowAxis 0
#define colAxis 1
#define channelAxis 2

// used by the decoder to allow two stage decoding
// read the header, then decode the data
typedef struct {
	png_structp png;
	png_infop   info;
	BufferPos   buffer;
} PngContext;

// this is used by the encoder to build a list of output buffers
// these are freed at the end of the encode
typedef struct BufferItem {
	unsigned char* data;
	size_t size;
	size_t pos;
	struct BufferItem* next;
	struct BufferItem* last;
	struct BufferItem* allocatedSelf;
} BufferItem;

//==============================================================================
// freeCByteArray
void freeCByteArray(CByteArray array) {
	if(array.data != NULL) free((void*)array.data);
}

//==============================================================================
// freePngCDecoderInfo
void freePngCDecoderInfo(CDecoderInfo info) {
	PngContext* context = (PngContext*)info.context;
	png_destroy_read_struct(&context->png, &context->info, NULL);
	free(context);
	free((void*)info.dims);
}

//==============================================================================
// accumulateAndFree
unsigned char* accumulateAndFree(BufferItem* item, CByteArray* result) {
	assert(item != NULL && result != NULL);
	unsigned char* pOut;
	result->count += item->pos;
	
	if (item->next != NULL) {
		pOut = accumulateAndFree(item->next, result);
		pOut -= item->pos;
		memcpy(pOut, item->data, item->pos);
	} else {
		result->data = malloc(result->count);
		pOut = (unsigned char*)result->data + (result->count - item->pos);
		memcpy(pOut, item->data, item->pos);
	}
	free(item->data);
	if (item->allocatedSelf) free(item->allocatedSelf);
	return pOut;
}
														 
//==============================================================================
// writeFn
void writeFn(png_structp png, png_bytep data, png_size_t count) {
	BufferItem* pRoot = (BufferItem* )png_get_io_ptr(png);
	BufferItem* pLast = pRoot->last;

	// find available space
	size_t available = pLast->size - pLast->pos;
	
	if (count <= available) {
		memcpy(&pLast->data[pLast->pos], data, count);
		pLast->pos += count;
	} else {
		// take as much as we can and advance the pos
		memcpy(&pLast->data[pLast->pos], data, available);
		pLast->pos = pLast->size;

		// update count and position
		count -= available;
		data  += available;
		
		// allocate a new item and copy the remainder
		BufferItem* next = (BufferItem*)malloc(sizeof(BufferItem));
		next->allocatedSelf = next;
		
		// estimate buffer size: TODO: test with chaining many small buffers
		size_t countPlusMargin = count + 1024;
		size_t lastSize  = pLast->size * .5;
		next->size = countPlusMargin > lastSize ? countPlusMargin : lastSize;
		if (next->size < 1024) next->size = 1024;
		
		next->pos = 0;
		next->data = malloc(next->size);

		next->last = NULL;
		next->next = NULL;
		pLast->next = next;
		pRoot->last = next;
		
		// call recursively until all of the data is consumed
		writeFn(png, data, count);
	}
}

//==============================================================================
// pngEncode
CByteArray pngEncode(const unsigned char* data, size_t elementCount,
										 const size_t* dims, size_t numDims, CDataType dataType,
										 long compression) {
	CByteArray encoded;
	encoded.count = 0;
	encoded.data = NULL;
	
	// linked list of buffers with the root item on the stack
	BufferItem rootBufferItem;
	memset(&rootBufferItem, 0, sizeof(rootBufferItem));
	rootBufferItem.last = &rootBufferItem;

	unsigned rows = (unsigned)dims[rowAxis];
	unsigned cols = (unsigned)dims[colAxis];
	unsigned channels = (unsigned)dims[channelAxis];
	
	// setup
	png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING,
																					  NULL, NULL, NULL);
	if(png == NULL) return encoded;
	
	png_infop info = png_create_info_struct(png);
	if(info == NULL) {
		png_destroy_write_struct(&png, NULL);
		return encoded;
	}
	
	if(setjmp(png_jmpbuf(png))) {
		png_destroy_write_struct(&png, &info);
		return encoded;
	}

	png_set_write_fn(png, &rootBufferItem, writeFn, NULL);

	// compression
	if (compression == -1) compression = Z_DEFAULT_COMPRESSION;
	png_set_compression_level(png, (int)compression);
	
	// TODO: make dynamic estimate
	double estimatedCompression = 0.5;
	
	// adjust based on target format
	int color_type = 0;
	switch (channels) {
		case 1:
			color_type = PNG_COLOR_TYPE_GRAY;
			break;
		case 2:
			color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
			break;
		case 3:
			color_type = PNG_COLOR_TYPE_RGB;
			break;
		case 4:
			color_type = PNG_COLOR_TYPE_RGBA;
			break;
			
		// you must specify an output type
		case CImageType_any:
		default: assert(0);
	}
	
	// set header info
	int bitDepth = dataType == CDataType_real8U ? 8 : 16;
	png_set_IHDR(png, info, cols, rows, bitDepth, color_type,
							 PNG_INTERLACE_NONE,
							 PNG_COMPRESSION_TYPE_DEFAULT,
							 PNG_FILTER_TYPE_DEFAULT);
	
	// allocate output buffer (min 1K)
	size_t bytesPerRow = png_get_rowbytes(png, info);
	rootBufferItem.size = ((bytesPerRow * rows) * estimatedCompression);
	if (rootBufferItem.size < 1024) rootBufferItem.size = 1024;
	rootBufferItem.data = malloc(rootBufferItem.size);

	// compression - the default seems to be fastest and smallest
//	png_set_compression_level(png, Z_BEST_SPEED);
	
	// write header
	png_write_info(png, info);

	// allocate index table and set pointers
	png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * rows);
	
	const unsigned char* pos = data;
	for(int row = 0; row < rows; row++, pos += bytesPerRow) {
		row_pointers[row] = (png_byte*)pos;
	}
	
	png_write_image(png, row_pointers);
	png_write_end(png, info);
	free(row_pointers);
	png_destroy_write_struct(&png, &info);
	
	// simple one buffer case
	if (rootBufferItem.next == NULL) {
		// ownership of the root buffer data is given to the encoded structure
		// which the caller must free
		encoded.data  = rootBufferItem.data;
		encoded.count = rootBufferItem.pos;
	} else {
		accumulateAndFree(&rootBufferItem, &encoded);
	}
	
	return encoded;
}

//==============================================================================
// readFn
void readFn(png_structp png, png_bytep outBytes, png_size_t byteCountToRead) {
	BufferPos* bufferPos = (BufferPos*)png_get_io_ptr(png);
	const unsigned char* start = bufferPos->data + bufferPos->pos;
	assert(start + byteCountToRead <= bufferPos->data + bufferPos->count);
	memcpy(outBytes, start, byteCountToRead);
	bufferPos->pos += byteCountToRead;
}

//==============================================================================
// pngDecodeInfo
CDecoderInfo pngDecodeInfo(const unsigned char* src, size_t count,
													 CImageType outImageType) {
	// context
	CDecoderInfo info;
	memset(&info, 0, sizeof(info));
	PngContext* context = malloc(sizeof(PngContext));
	memset(context, 0, sizeof(PngContext));
	info.context = context;
	info.numDims = 3;
	info.dims = malloc(info.numDims * sizeof(size_t));
	info.status = CStatus_success;
	info.willChangeType = 1;

	// this is to support the readFn
	context->buffer.data  = src;
	context->buffer.count = count;
	context->buffer.pos   = 0;
	
	// create read struct
	png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING,
																					 NULL, NULL, NULL);
	context->png = png;
	
	if(png == NULL) {
		info.status = CStatus_error;
		return info;
	}
	
	png_infop png_info = png_create_info_struct(png);
	context->info = png_info;
	
	if(png_info == NULL) {
		png_destroy_read_struct(&png, NULL, NULL);
		info.status = CStatus_error;
		return info;
	}
	
	if(setjmp(png_jmpbuf(png))) {
		png_destroy_read_struct(&png, &png_info, NULL);
		info.status = CStatus_error;
		return info;
	}
	
	// read the header
	png_set_read_fn(png, &context->buffer, readFn);
	png_read_info(png, png_info);
	
	int bit_depth = png_get_bit_depth(png, png_info);
	
	png_byte color_type = png_get_color_type(png, png_info);
	if(bit_depth < 8)	png_set_expand_gray_1_2_4_to_8(png);

	// set adjustments to match desired output format
	switch (outImageType) {
  case CImageType_gray:
		switch (color_type) {
			case PNG_COLOR_TYPE_GRAY:
				info.willChangeType = 0;
				break;
			case PNG_COLOR_TYPE_GRAY_ALPHA:
				png_set_strip_alpha(png);
				break;
			case PNG_COLOR_TYPE_PALETTE:
				png_set_expand(png);
				if(png_get_valid(png, png_info, PNG_INFO_tRNS))
					png_set_tRNS_to_alpha(png);
				break;
			case PNG_COLOR_TYPE_RGB:
				png_set_rgb_to_gray(png, 1, -1, -1);
				break;
				
			case PNG_COLOR_TYPE_RGBA:
				png_set_rgb_to_gray(png, 1, -1, -1);
				png_set_strip_alpha(png);
				break;
			default: abort();
		}
		break;
			
  case CImageType_grayAlpha:
		switch (color_type) {
			case PNG_COLOR_TYPE_GRAY:
				png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
				break;
			case PNG_COLOR_TYPE_GRAY_ALPHA:
				info.willChangeType = 0;
				break;
			case PNG_COLOR_TYPE_PALETTE:
				png_set_expand(png);
				if(png_get_valid(png, png_info, PNG_INFO_tRNS))
					png_set_tRNS_to_alpha(png);
				png_set_rgb_to_gray(png, 1, -1, -1);
				png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
				break;
			case PNG_COLOR_TYPE_RGB:
				png_set_rgb_to_gray(png, 1, -1, -1);
				png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
				break;
			case PNG_COLOR_TYPE_RGBA:
				png_set_rgb_to_gray(png, 1, -1, -1);
				break;
			default: abort();
		}
		break;
			
  case CImageType_rgb:
		switch (color_type) {
			case PNG_COLOR_TYPE_GRAY:
				png_set_gray_to_rgb(png);
				break;
			case PNG_COLOR_TYPE_GRAY_ALPHA:
				png_set_gray_to_rgb(png);
				png_set_strip_alpha(png);
				break;
			case PNG_COLOR_TYPE_PALETTE:
				png_set_expand(png);
				if(png_get_valid(png, png_info, PNG_INFO_tRNS))
					png_set_tRNS_to_alpha(png);
				break;
			case PNG_COLOR_TYPE_RGB:
				info.willChangeType = 0;
				break;
			case PNG_COLOR_TYPE_RGBA:
				png_set_strip_alpha(png);
				break;
			default: abort();
		}
		break;
			
  case CImageType_rgba:
		switch (color_type) {
			case PNG_COLOR_TYPE_GRAY:
				png_set_gray_to_rgb(png);
				png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
				break;
			case PNG_COLOR_TYPE_GRAY_ALPHA:
				png_set_gray_to_rgb(png);
				png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
				break;
			case PNG_COLOR_TYPE_PALETTE:
				png_set_expand(png);
				if(png_get_valid(png, png_info, PNG_INFO_tRNS))
					png_set_tRNS_to_alpha(png);
				png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
				break;
			case PNG_COLOR_TYPE_RGB:
				png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
				break;
			case PNG_COLOR_TYPE_RGBA:
				info.willChangeType = 0;
				break;
			default: abort();
		}
		break;
			
  case CImageType_any:
			info.willChangeType = 0;
			break;
	}
	
	png_read_update_info(png, png_info);
	
	// bit depth
	bit_depth = png_get_bit_depth(png, png_info);
	info.dataTypeSize = bit_depth > 8 ? 2 : 1;
	info.dataType = info.dataTypeSize == 1 ? CDataType_real8U : CDataType_real16U;
	
	// cast away const so we can set the return value
	size_t* dims = (size_t *)info.dims;
	dims[rowAxis] = png_get_image_height(png, png_info);
	dims[colAxis] = png_get_image_width(png, png_info);
	dims[channelAxis] = png_get_channels(png, png_info);
	
	// allocation requirements
	info.byteCount = info.dims[rowAxis] * info.dims[colAxis] *
	                 info.dims[channelAxis] * info.dataTypeSize;
	assert(png_get_rowbytes(png, png_info) * info.dims[rowAxis] == info.byteCount);
	return info;
}

//==============================================================================
// pngDecodeData
CStatus pngDecodeData(CDecoderInfo info, unsigned char* dst, size_t dstByteCount) {

	// validate
	assert(dst != NULL);
	assert(dstByteCount == info.byteCount);
	PngContext* context = (PngContext*)info.context;
	size_t rows = info.dims[rowAxis];
	size_t cols = info.dims[colAxis];
	size_t channels = info.dims[channelAxis];
	size_t bytesPerRow = channels * cols * info.dataTypeSize;
	assert(bytesPerRow == png_get_rowbytes(context->png, context->info));
	
	// allocate index table and set pointers
	png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * rows);
	
	if (row_pointers != NULL) {
		const unsigned char* pos = dst;
		for(int row = 0; row < rows; row++, pos += bytesPerRow) {
			row_pointers[row] = (png_byte*)pos;
		}
		
		// read the image into destination memory
		png_read_image(context->png, row_pointers);
		free(row_pointers);
	}
	
	return  CStatus_success;
}
