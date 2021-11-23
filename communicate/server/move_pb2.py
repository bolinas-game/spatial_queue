# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: move.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='move.proto',
  package='protomove',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\nmove.proto\x12\tprotomove\"\"\n\tDirection\x12\n\n\x02op\x18\x01 \x01(\t\x12\t\n\x01t\x18\x02 \x01(\x05\"=\n\x0b\x42riefPoints\x12\n\n\x02x1\x18\x01 \x01(\x02\x12\n\n\x02y1\x18\x02 \x01(\x02\x12\n\n\x02x2\x18\x03 \x01(\x02\x12\n\n\x02y2\x18\x04 \x01(\x02\"]\n\x05Point\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0b\n\x03lon\x18\x02 \x01(\x02\x12\x0b\n\x03lat\x18\x03 \x01(\x02\x12\t\n\x01x\x18\x04 \x01(\x02\x12\t\n\x01y\x18\x05 \x01(\x02\x12\r\n\x05\x61ngle\x18\x06 \x01(\x02\x12\t\n\x01t\x18\x07 \x01(\x05\"%\n\x08StopInfo\x12\x0e\n\x06isStop\x18\x01 \x01(\x08\x12\t\n\x01t\x18\x02 \x01(\x05\"&\n\nChangeInfo\x12\r\n\x05image\x18\x01 \x01(\t\x12\t\n\x01t\x18\x02 \x01(\x05\"+\n\x0c\x43ompleteInfo\x12\x10\n\x08\x63omplete\x18\x01 \x01(\x08\x12\t\n\x01t\x18\x02 \x01(\x05\"v\n\rStreamRequest\x12(\n\x08\x64ir_info\x18\x01 \x01(\x0b\x32\x14.protomove.DirectionH\x00\x12)\n\x07se_info\x18\x02 \x01(\x0b\x32\x16.protomove.BriefPointsH\x00\x42\x10\n\x0estream_request\"\x84\x02\n\x0eStreamResponse\x12&\n\npoint_info\x18\x01 \x01(\x0b\x32\x10.protomove.PointH\x00\x12(\n\tstop_info\x18\x02 \x01(\x0b\x32\x13.protomove.StopInfoH\x00\x12,\n\x0b\x63hange_info\x18\x03 \x01(\x0b\x32\x15.protomove.ChangeInfoH\x00\x12\x30\n\rcomplete_info\x18\x04 \x01(\x0b\x32\x17.protomove.CompleteInfoH\x00\x12-\n\x0bnew_se_info\x18\x05 \x01(\x0b\x32\x16.protomove.BriefPointsH\x00\x42\x11\n\x0fstream_response\"\x17\n\x06Status\x12\r\n\x05state\x18\x01 \x01(\t\"\x14\n\x03\x45nd\x12\r\n\x05state\x18\x01 \x01(\t2\x84\x01\n\x04Move\x12H\n\x0bGetPosition\x12\x18.protomove.StreamRequest\x1a\x19.protomove.StreamResponse\"\x00(\x01\x30\x01\x12\x32\n\x0bUnSubscribe\x12\x0e.protomove.End\x1a\x11.protomove.Status\"\x00\x62\x06proto3'
)




_DIRECTION = _descriptor.Descriptor(
  name='Direction',
  full_name='protomove.Direction',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='op', full_name='protomove.Direction.op', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='t', full_name='protomove.Direction.t', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=25,
  serialized_end=59,
)


_BRIEFPOINTS = _descriptor.Descriptor(
  name='BriefPoints',
  full_name='protomove.BriefPoints',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='x1', full_name='protomove.BriefPoints.x1', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='y1', full_name='protomove.BriefPoints.y1', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='x2', full_name='protomove.BriefPoints.x2', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='y2', full_name='protomove.BriefPoints.y2', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=61,
  serialized_end=122,
)


_POINT = _descriptor.Descriptor(
  name='Point',
  full_name='protomove.Point',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='protomove.Point.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='lon', full_name='protomove.Point.lon', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='lat', full_name='protomove.Point.lat', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='x', full_name='protomove.Point.x', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='y', full_name='protomove.Point.y', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='angle', full_name='protomove.Point.angle', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='t', full_name='protomove.Point.t', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=124,
  serialized_end=217,
)


_STOPINFO = _descriptor.Descriptor(
  name='StopInfo',
  full_name='protomove.StopInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='isStop', full_name='protomove.StopInfo.isStop', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='t', full_name='protomove.StopInfo.t', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=219,
  serialized_end=256,
)


_CHANGEINFO = _descriptor.Descriptor(
  name='ChangeInfo',
  full_name='protomove.ChangeInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='image', full_name='protomove.ChangeInfo.image', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='t', full_name='protomove.ChangeInfo.t', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=258,
  serialized_end=296,
)


_COMPLETEINFO = _descriptor.Descriptor(
  name='CompleteInfo',
  full_name='protomove.CompleteInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='complete', full_name='protomove.CompleteInfo.complete', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='t', full_name='protomove.CompleteInfo.t', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=298,
  serialized_end=341,
)


_STREAMREQUEST = _descriptor.Descriptor(
  name='StreamRequest',
  full_name='protomove.StreamRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='dir_info', full_name='protomove.StreamRequest.dir_info', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='se_info', full_name='protomove.StreamRequest.se_info', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='stream_request', full_name='protomove.StreamRequest.stream_request',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=343,
  serialized_end=461,
)


_STREAMRESPONSE = _descriptor.Descriptor(
  name='StreamResponse',
  full_name='protomove.StreamResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='point_info', full_name='protomove.StreamResponse.point_info', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='stop_info', full_name='protomove.StreamResponse.stop_info', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='change_info', full_name='protomove.StreamResponse.change_info', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='complete_info', full_name='protomove.StreamResponse.complete_info', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='new_se_info', full_name='protomove.StreamResponse.new_se_info', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='stream_response', full_name='protomove.StreamResponse.stream_response',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=464,
  serialized_end=724,
)


_STATUS = _descriptor.Descriptor(
  name='Status',
  full_name='protomove.Status',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='state', full_name='protomove.Status.state', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=726,
  serialized_end=749,
)


_END = _descriptor.Descriptor(
  name='End',
  full_name='protomove.End',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='state', full_name='protomove.End.state', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=751,
  serialized_end=771,
)

_STREAMREQUEST.fields_by_name['dir_info'].message_type = _DIRECTION
_STREAMREQUEST.fields_by_name['se_info'].message_type = _BRIEFPOINTS
_STREAMREQUEST.oneofs_by_name['stream_request'].fields.append(
  _STREAMREQUEST.fields_by_name['dir_info'])
_STREAMREQUEST.fields_by_name['dir_info'].containing_oneof = _STREAMREQUEST.oneofs_by_name['stream_request']
_STREAMREQUEST.oneofs_by_name['stream_request'].fields.append(
  _STREAMREQUEST.fields_by_name['se_info'])
_STREAMREQUEST.fields_by_name['se_info'].containing_oneof = _STREAMREQUEST.oneofs_by_name['stream_request']
_STREAMRESPONSE.fields_by_name['point_info'].message_type = _POINT
_STREAMRESPONSE.fields_by_name['stop_info'].message_type = _STOPINFO
_STREAMRESPONSE.fields_by_name['change_info'].message_type = _CHANGEINFO
_STREAMRESPONSE.fields_by_name['complete_info'].message_type = _COMPLETEINFO
_STREAMRESPONSE.fields_by_name['new_se_info'].message_type = _BRIEFPOINTS
_STREAMRESPONSE.oneofs_by_name['stream_response'].fields.append(
  _STREAMRESPONSE.fields_by_name['point_info'])
_STREAMRESPONSE.fields_by_name['point_info'].containing_oneof = _STREAMRESPONSE.oneofs_by_name['stream_response']
_STREAMRESPONSE.oneofs_by_name['stream_response'].fields.append(
  _STREAMRESPONSE.fields_by_name['stop_info'])
_STREAMRESPONSE.fields_by_name['stop_info'].containing_oneof = _STREAMRESPONSE.oneofs_by_name['stream_response']
_STREAMRESPONSE.oneofs_by_name['stream_response'].fields.append(
  _STREAMRESPONSE.fields_by_name['change_info'])
_STREAMRESPONSE.fields_by_name['change_info'].containing_oneof = _STREAMRESPONSE.oneofs_by_name['stream_response']
_STREAMRESPONSE.oneofs_by_name['stream_response'].fields.append(
  _STREAMRESPONSE.fields_by_name['complete_info'])
_STREAMRESPONSE.fields_by_name['complete_info'].containing_oneof = _STREAMRESPONSE.oneofs_by_name['stream_response']
_STREAMRESPONSE.oneofs_by_name['stream_response'].fields.append(
  _STREAMRESPONSE.fields_by_name['new_se_info'])
_STREAMRESPONSE.fields_by_name['new_se_info'].containing_oneof = _STREAMRESPONSE.oneofs_by_name['stream_response']
DESCRIPTOR.message_types_by_name['Direction'] = _DIRECTION
DESCRIPTOR.message_types_by_name['BriefPoints'] = _BRIEFPOINTS
DESCRIPTOR.message_types_by_name['Point'] = _POINT
DESCRIPTOR.message_types_by_name['StopInfo'] = _STOPINFO
DESCRIPTOR.message_types_by_name['ChangeInfo'] = _CHANGEINFO
DESCRIPTOR.message_types_by_name['CompleteInfo'] = _COMPLETEINFO
DESCRIPTOR.message_types_by_name['StreamRequest'] = _STREAMREQUEST
DESCRIPTOR.message_types_by_name['StreamResponse'] = _STREAMRESPONSE
DESCRIPTOR.message_types_by_name['Status'] = _STATUS
DESCRIPTOR.message_types_by_name['End'] = _END
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Direction = _reflection.GeneratedProtocolMessageType('Direction', (_message.Message,), {
  'DESCRIPTOR' : _DIRECTION,
  '__module__' : 'move_pb2'
  # @@protoc_insertion_point(class_scope:protomove.Direction)
  })
_sym_db.RegisterMessage(Direction)

BriefPoints = _reflection.GeneratedProtocolMessageType('BriefPoints', (_message.Message,), {
  'DESCRIPTOR' : _BRIEFPOINTS,
  '__module__' : 'move_pb2'
  # @@protoc_insertion_point(class_scope:protomove.BriefPoints)
  })
_sym_db.RegisterMessage(BriefPoints)

Point = _reflection.GeneratedProtocolMessageType('Point', (_message.Message,), {
  'DESCRIPTOR' : _POINT,
  '__module__' : 'move_pb2'
  # @@protoc_insertion_point(class_scope:protomove.Point)
  })
_sym_db.RegisterMessage(Point)

StopInfo = _reflection.GeneratedProtocolMessageType('StopInfo', (_message.Message,), {
  'DESCRIPTOR' : _STOPINFO,
  '__module__' : 'move_pb2'
  # @@protoc_insertion_point(class_scope:protomove.StopInfo)
  })
_sym_db.RegisterMessage(StopInfo)

ChangeInfo = _reflection.GeneratedProtocolMessageType('ChangeInfo', (_message.Message,), {
  'DESCRIPTOR' : _CHANGEINFO,
  '__module__' : 'move_pb2'
  # @@protoc_insertion_point(class_scope:protomove.ChangeInfo)
  })
_sym_db.RegisterMessage(ChangeInfo)

CompleteInfo = _reflection.GeneratedProtocolMessageType('CompleteInfo', (_message.Message,), {
  'DESCRIPTOR' : _COMPLETEINFO,
  '__module__' : 'move_pb2'
  # @@protoc_insertion_point(class_scope:protomove.CompleteInfo)
  })
_sym_db.RegisterMessage(CompleteInfo)

StreamRequest = _reflection.GeneratedProtocolMessageType('StreamRequest', (_message.Message,), {
  'DESCRIPTOR' : _STREAMREQUEST,
  '__module__' : 'move_pb2'
  # @@protoc_insertion_point(class_scope:protomove.StreamRequest)
  })
_sym_db.RegisterMessage(StreamRequest)

StreamResponse = _reflection.GeneratedProtocolMessageType('StreamResponse', (_message.Message,), {
  'DESCRIPTOR' : _STREAMRESPONSE,
  '__module__' : 'move_pb2'
  # @@protoc_insertion_point(class_scope:protomove.StreamResponse)
  })
_sym_db.RegisterMessage(StreamResponse)

Status = _reflection.GeneratedProtocolMessageType('Status', (_message.Message,), {
  'DESCRIPTOR' : _STATUS,
  '__module__' : 'move_pb2'
  # @@protoc_insertion_point(class_scope:protomove.Status)
  })
_sym_db.RegisterMessage(Status)

End = _reflection.GeneratedProtocolMessageType('End', (_message.Message,), {
  'DESCRIPTOR' : _END,
  '__module__' : 'move_pb2'
  # @@protoc_insertion_point(class_scope:protomove.End)
  })
_sym_db.RegisterMessage(End)



_MOVE = _descriptor.ServiceDescriptor(
  name='Move',
  full_name='protomove.Move',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=774,
  serialized_end=906,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetPosition',
    full_name='protomove.Move.GetPosition',
    index=0,
    containing_service=None,
    input_type=_STREAMREQUEST,
    output_type=_STREAMRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='UnSubscribe',
    full_name='protomove.Move.UnSubscribe',
    index=1,
    containing_service=None,
    input_type=_END,
    output_type=_STATUS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_MOVE)

DESCRIPTOR.services_by_name['Move'] = _MOVE

# @@protoc_insertion_point(module_scope)
