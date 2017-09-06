//******************************************************************************
//  Created by Edward Connell on 9/28/16.
//  Copyright © 2016 Connell Research. All rights reserved.
//
//  https://www4.cs.fau.de/Services/Doc/graphics/doc/jpeg/libjpeg.html
//
#include "include/ImageCodecs.h"
#include <stdio.h>
#include <jpeglib.h>
#include <memory.h>
#include <assert.h>
#include <setjmp.h>

// axes
#define rowAxis 0
#define colAxis 1
#define channelAxis 2

//==============================================================================
// ErrorMgr
struct ErrorMgr {
	struct jpeg_error_mgr pub;
	jmp_buf setjmp_buffer;	/* for return to caller */
};

typedef struct ErrorMgr* ErrorMgrPtr;

// routine that will replace the standard error_exit method:
void onErrorExit (j_common_ptr cinfo)
{
	/* cinfo->err really points to a ErrorMgr struct, so coerce pointer */
	ErrorMgrPtr myerr = (ErrorMgrPtr) cinfo->err;
	
	/* Always display the message. */
	/* We could postpone this until after returning, if we chose. */
	(*cinfo->err->output_message) (cinfo);
	
	/* Return control to the setjmp point */
	longjmp(myerr->setjmp_buffer, 1);
}

// used by the decoder to allow two stage decoding
// read the header, then decode the data
typedef struct {
	struct jpeg_decompress_struct cinfo;
	struct ErrorMgr jerr;
} JpegContext;

//==============================================================================
// freeJpegCDecoderInfo
void freeJpegCDecoderInfo(CDecoderInfo info) {
	JpegContext* context = (JpegContext*)info.context;
	jpeg_destroy_decompress(&context->cinfo);
	free(context);
	free((void*)info.dims);
}

//==============================================================================
// jpegEncode
CByteArray jpegEncode(const unsigned char* data, size_t elementCount,
										  const size_t* dims, size_t numDims, size_t quality) {
	// validate types
	int rows = (int)dims[rowAxis];
	int cols = (int)dims[colAxis];
	int channels = (int)dims[channelAxis];
	assert(channels == 1 || channels == 3 || channels == 4);
	
	CByteArray encoded;
	encoded.count = 0;
	encoded.data = NULL;

	// initialize JPEG compression object
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	
	jpeg_create_compress(&cinfo);

	// specify data destination
	jpeg_mem_dest(&cinfo, (unsigned char **)&encoded.data, &encoded.count);
	
	// set parameters
	cinfo.image_height     = rows;
	cinfo.image_width      = cols;
	cinfo.input_components = channels;
	cinfo.in_color_space   = channels == 1 ? JCS_GRAYSCALE : JCS_RGB;

	// use the library's routine to set default compression parameters
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, (int)quality, TRUE);
	
	// TRUE ensures to write a complete interchange-JPEG file
	jpeg_start_compress(&cinfo, TRUE);
	
	// stride and pointer to JSAMPLE row[s]
	int row_stride = cinfo.image_width * cinfo.input_components;
	JSAMPROW row_pointer[1];
	
	// loop through the scan lines
	while (cinfo.next_scanline < cinfo.image_height) {
		row_pointer[0] = (unsigned char*)&data[cinfo.next_scanline * row_stride];
		(void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
	}
	
	// clean up
	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);
	
	return encoded;
}

//==============================================================================
// jpegDecodeInfo
CDecoderInfo jpegDecodeInfo(const unsigned char* src, size_t count,
														CImageType outImageType) {
	// adjust output type
	if (outImageType == CImageType_grayAlpha) outImageType = CImageType_gray;
	if (outImageType == CImageType_rgba) outImageType = CImageType_rgb;
	
	// context
	CDecoderInfo info;
	JpegContext* context = malloc(sizeof(JpegContext));
	memset(context, 0, sizeof(JpegContext));
	info.context = context;
	info.numDims = 3;
	info.dims = malloc(info.numDims * sizeof(size_t));
	info.status = CStatus_success;
	info.dataType = CDataType_real8U;
	info.dataTypeSize = 1;
	
	// setup custom error exit routine
	context->cinfo.err = jpeg_std_error(&context->jerr.pub);
	context->jerr.pub.error_exit = onErrorExit;
	
	if (setjmp(context->jerr.setjmp_buffer)) {
		jpeg_destroy_decompress(&context->cinfo);
		info.status = CStatus_error;
		return info;
	}
	
	// initialize the JPEG decompression object
	jpeg_create_decompress(&context->cinfo);
	jpeg_mem_src(&context->cinfo, (unsigned char *)src, count);
	
	// read header info
	jpeg_read_header(&context->cinfo, TRUE);
	size_t* dims = (size_t*)info.dims;
	dims[rowAxis] = context->cinfo.image_height;
	dims[colAxis] = context->cinfo.image_width;
	dims[channelAxis] = context->cinfo.num_components;
	
	// set parameters for decompression
	info.willChangeType = 0;
	switch (outImageType) {
		case CImageType_gray:
			if (context->cinfo.out_color_space != JCS_GRAYSCALE) {
				context->cinfo.out_color_space = JCS_GRAYSCALE;
				dims[channelAxis] = 1;
				info.willChangeType = 1;
			}
			break;
			
		case CImageType_rgb:
			if (context->cinfo.out_color_space != JCS_RGB) {
				context->cinfo.out_color_space = JCS_RGB;
				dims[channelAxis] = 3;
				info.willChangeType = 1;
			}
			break;
			
		default: break;
	}

	return info;
}

//==============================================================================
// jpegDecodeData
// https://dev.w3.org/Amaya/libjpeg/jdapimin.c
//
CStatus jpegDecodeData(CDecoderInfo info, unsigned char* dst, size_t dstByteCount) {
	// start decompressor
	JpegContext* context = (JpegContext*)info.context;
	jpeg_start_decompress(&context->cinfo);
	
	// create make an output work buffer of the right size.
	int bytesPerRow = context->cinfo.output_width * context->cinfo.output_components;
	assert(bytesPerRow * context->cinfo.output_height == dstByteCount);
	
	while (context->cinfo.output_scanline < context->cinfo.output_height)
	{
		jpeg_read_scanlines(&context->cinfo, &dst, 1);
		dst += bytesPerRow;
	}
	jpeg_finish_decompress(&context->cinfo);
	
	return (context->jerr.pub.num_warnings > 0) ? CStatus_error : CStatus_success;
}

