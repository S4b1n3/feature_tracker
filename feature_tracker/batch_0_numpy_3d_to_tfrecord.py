# -*- coding: utf-8 -*-
"""
# TFRecords writer code for Numpy array data (specifically for 3D arrays with an additional channel dimension),
# the interpret_npy_header function can be applied for any type of numpy array.

# The source code is partially based on Google Inc's build_imagenet_data.py:
# https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

#######################################################################################################
####    RUNNING INSTRUCTIONS:
####    Run simply with python compressed_hmdb_features_numpy3d_to_tfrecords.py
####    Before running, do the following:
####         1. change the paths for input and output directories.
####         2. in features, toggle the feature 'nframes'. Use it for test records, and not for train.
####         3. add batch number -batch_#-
####         4. at the bottom, change for test or train.
#######################################################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import six
import tensorflow as tf

def interpret_npy_header(encoded_image_data):
    """Extracts numpy header information from byte encoded .npy files
    
    Args:
        encoded_image_data: string, (bytes) representation of a numpy array as the tf.gfile.FastGFile.read() method returns it
    Returns:
        header_len: integer, length of header information in bytes
        dt: numpy datatype with correct byte order (little or bigEndian)
        index_order: character 'C' for C-style indexing, 'F' for Fortran-style indexing
        np_array_shape: numpy array, original shape of the encoded numpy array
    """
    #Check if the encoded data is a numpy array or not
    numpy_prefix = b'\x93NUMPY'
    if encoded_image_data[:6] != numpy_prefix:
        raise ValueError('The encoded data is not a numpy array')
    
    #Check if the encoded data is not corrupted and long enough to hold the whole header information
    if len(encoded_image_data)>10:
        #header length in bytes is encoded in the 8-9th bytes of the data as an uint16 number
        header_len=np.frombuffer(encoded_image_data, dtype=np.uint16, offset=8,count=1)[0]
        #extract header data based on variable header length
        header_data=str(encoded_image_data[10:10+header_len])
        
        #extract data type information from the header
        dtypes_dict = {'u1': np.uint8, 'u2': np.uint16, 'u4' : np.uint32, 'u8': np.uint64,
                       'i1': np.int8, 'i2': np.int16, 'i4': np.int32, 'i8': np.int64,
                       'f4': np.float32, 'f8': np.float64, 'b1': bool}

        start_datatype = header_data.find("'descr': ")+10
        dt = dtypes_dict[header_data[start_datatype+1:start_datatype+3]]
        
        #both big and littleEndian byte order should be interpreted correctly
        if header_data[start_datatype:start_datatype+1] is '>':
            dt = dt.newbyteorder('>')
            
        
        #extract index ordering information from the header
        index_order='C'
        start_index_order=header_data.find("'fortran_order': ")+17
        if header_data[start_index_order:start_index_order+4]=='True':
            index_order='F'
        
        
        #extract array shape from the header
        start_shape = header_data.find("'shape': (")+10
        end_shape = start_shape + header_data[start_shape:].find(")")
        
        np_array_shape=np.fromstring(header_data[start_shape:end_shape],dtype=int, sep=',')
        
        return header_len,dt,index_order,np_array_shape
    else:
        raise ValueError('The encoded data length is not sufficient')
        

#define input and output data directory
tf.compat.v1.flags.DEFINE_string('input_directory', './10dist_control_phases/32frames/batch_5/pathtracker/samples/','Input data directory')
tf.compat.v1.flags.DEFINE_string('output_directory', './10dist_control_phases/32frames/pathtracker/tfrecords/','Output data directory')

#define how many training a validatin (and test) TFRecord files you want to write
tf.compat.v1.flags.DEFINE_integer('train_shards', 40, 'Number of shards in training TFRecord files.') #2
tf.compat.v1.flags.DEFINE_integer('validation_shards', 4, 'Number of shards in validation TFRecord files.') #1

#define the number of threads (validation and train shards have to be integer multiples of the number of threads)
tf.compat.v1.flags.DEFINE_integer('num_threads', 4, 'Number of threads to preprocess the images.') #1

#define paths to train, validation and test label files
# ***
# tf.compat.v1.flags.DEFINE_string('train_labels_file', './input/train_labels.csv', 'Labels file')
# tf.compat.v1.flags.DEFINE_string('validation_labels_file', './input/validation_labels.csv', 'Labels file')
# tf.compat.v1.flags.DEFINE_string('test_labels_file', './input/test_labels.csv', 'Labels file')

FLAGS = tf.compat.v1.flags.FLAGS


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if isinstance(value, six.string_types):           
    value = six.binary_type(value, encoding='utf-8') 
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, height, width, depth, num_channels):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.npy'
    image_buffer: string, encoded 3D numpy array
    label: integer, identifier for the ground truth for the network
    height: integer, image height in voxels
    width: integer, image width in voxels
    depth: integer, image depth in voxels
    num_channels: integer, number of color channels (last dimension of the numpy array)
  Returns:
    Example proto
  """
  
  #determine colorspace from the number of channels  
  colorspace=""
  if num_channels == 1:
      colorspace = 'grayscale'
  if num_channels == 3:
      colorspace = 'RGB'
  

  example = tf.train.Example(features=tf.train.Features(feature={
      'height': _int64_feature(height),
      'width': _int64_feature(width),
    #   'depth': _int64_feature(depth),
      # 'image/colorspace': _bytes_feature(colorspace), # <- remove
    #   'channels': _int64_feature(num_channels),
      'label': _bytes_feature(np.uint8(label).tostring()),
    #   'nframes': _int64_feature(depth), # <-- ***** USE IF GENERATING TEST SHARDS 

    #   'filename': _bytes_feature(os.path.basename(filename)),
      'image': _bytes_feature(image_buffer)}))
  return example


def _process_image(data_dir, filename):
  """Process a single image file.

  Args:
    data_dir: string, root directory of images eg. './data/'
    filename: string, path to an image file in numpy format e.g., 'example.npy'
  Returns:
    image_buffer: string, encoded numpy array (should be 3D + a channel dimension)
    height: integer, image height in voxels
    width: integer, image width in voxels
    depth: integer, image depth in voxels
    num_channels: integer, number of color channels (last dimension of the numpy array)
  """

  # Read the numpy array.
  with tf.compat.v1.gfile.FastGFile(data_dir+filename, 'rb') as f:
    image_data = f.read()
  
  try:
    #extract numpy header information
    header_len,dt,index_order,np_array_shape = interpret_npy_header(image_data)
    image = np.frombuffer(image_data, dtype=dt, offset=10+header_len)
    image = np.reshape(image,np_array_shape,order=index_order)
  except ValueError as err:
    print(err)

  # Check that the image is a 3D array + a channel dimension
  assert len(image.shape) == 5
  depth = image.shape[1]
  height = image.shape[2]
  width = image.shape[3]
  num_channels = image.shape[4]
  # print("#### image is of dim: "+str(height)+", "+str(width)+", "+str(depth)+", "+str(num_channels)+"\n")
    

  return image_data[10+header_len:], height, width, depth, num_channels 

def _process_image_files_batch(thread_index, ranges, name, filenames, labels, num_shards=1):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    
    options = tf.io.TFRecordOptions(compression_type = 'GZIP')

    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.io.TFRecordWriter(output_file, options=options)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      
      filename = filenames[i]
      label = labels[i]
      
      image_buffer, height, width, depth, num_channels = _process_image(data_dir=FLAGS.input_directory,filename=filename)

      example = _convert_to_example(filename, image_buffer, label, height, width, depth, num_channels)
      print("filename: %s label: %d" % (filename,label))
      
      writer.write(example.SerializeToString())
            
      shard_counter += 1
      counter += 1

      if not counter % 100:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()
  
def _process_image_files(name, filenames, labels, num_shards=1):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  
  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(int)
  ranges = []
  threads = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])
    
  # Launch a thread for each batch.
  print('Launching %d threads %d ranges with spacings: %s' % (FLAGS.num_threads,len(ranges), ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()
  
  threads = []
  for thread_index in range(FLAGS.num_threads):
    args = (thread_index, ranges, name, filenames, labels, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()
  
def _find_image_files(data_dir, labels_file=""):
  """Build a list of all images files and labels in the data set.

  Args:
    data_dir: string, path to the root directory of images.
    labels_file: csv with two columns: [0] contains filenames, [1] contains labels

  Returns:
    filenames: list of strings; each string is a path to an image file.
    labels: list of integer; each integer identifies the ground truth.
  """
  print('Determining list of input files and labels from %s.' % labels_file)
  
  # import pandas as pd
  # labels_df=pd.read_csv(labels_file)
  
  # filenames = list(labels_df['filenames'])      
  # labels = list(labels_df['labels'])
  import glob
  files=glob.glob(FLAGS.input_directory+"*.npy")
  filenames=[f.split("/")[-1] for f in files]
  labels=[int(f.split("_")[0]) for f in filenames]

  # print(files)
  # print(filenames)
  # print(labels)
 
  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]
   
  print('Found %d files inside %s.' %
        (len(filenames), data_dir))
  return filenames, labels

def _process_dataset(name, directory, num_shards, labels_file=""):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: path to a csv file containing filenames and labels
  """
  filenames, labels = _find_image_files(directory, labels_file)
   
  _process_image_files(name=name, filenames=filenames, labels=labels, num_shards=num_shards)

  
def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output_directory)
  
  # Defining batch number here, not making it a flag, simply because it might be null in few cases, and we do not want to have extra '-'s there. 
  batch_num='-batch_5-'
  
  # Run it for train, test and validation datasets
  # _process_dataset('validation', FLAGS.input_directory,FLAGS.validation_shards,FLAGS.validation_labels_file)
  # _process_dataset('test', FLAGS.input_directory,FLAGS.validation_shards,FLAGS.test_labels_file)
  _process_dataset('val'+batch_num, FLAGS.input_directory, FLAGS.train_shards)#,FLAGS.train_labels_file)
 
if __name__ == '__main__':
  tf.compat.v1.app.run()

