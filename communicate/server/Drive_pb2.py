# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Drive.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='Drive.proto',
  package='protodrive',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0b\x44rive.proto\x12\nprotodrive\"\"\n\tDirection\x12\n\n\x02op\x18\x01 \x01(\t\x12\t\n\x01t\x18\x02 \x01(\x05\"]\n\x05Point\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0b\n\x03lon\x18\x02 \x01(\x02\x12\x0b\n\x03lat\x18\x03 \x01(\x02\x12\t\n\x01x\x18\x04 \x01(\x02\x12\t\n\x01y\x18\x05 \x01(\x02\x12\r\n\x05\x61ngle\x18\x06 \x01(\x02\x12\t\n\x01t\x18\x07 \x01(\x05\"=\n\x0b\x42riefPoints\x12\n\n\x02x1\x18\x01 \x01(\x02\x12\n\n\x02y1\x18\x02 \x01(\x02\x12\n\n\x02x2\x18\x03 \x01(\x02\x12\n\n\x02y2\x18\x04 \x01(\x02\"%\n\x08StopInfo\x12\x0e\n\x06isStop\x18\x01 \x01(\x08\x12\t\n\x01t\x18\x02 \x01(\x05\"&\n\nChangeInfo\x12\r\n\x05image\x18\x01 \x01(\t\x12\t\n\x01t\x18\x02 \x01(\x05\"\x17\n\x06SEInfo\x12\r\n\x05state\x18\x01 \x01(\t\"\x18\n\x07\x45ndInfo\x12\r\n\x05state\x18\x01 \x01(\t2\xb9\x02\n\x05\x44rive\x12;\n\x0bGetPosition\x12\x15.protodrive.Direction\x1a\x11.protodrive.Point\"\x00\x30\x01\x12;\n\nDriverStop\x12\x15.protodrive.Direction\x1a\x14.protodrive.StopInfo\"\x00\x12>\n\x0b\x43hangeImage\x12\x15.protodrive.Direction\x1a\x16.protodrive.ChangeInfo\"\x00\x12?\n\x0eSendSEPosition\x12\x17.protodrive.BriefPoints\x1a\x12.protodrive.SEInfo\"\x00\x12\x35\n\x07SendEnd\x12\x13.protodrive.EndInfo\x1a\x13.protodrive.EndInfo\"\x00\x62\x06proto3'
)




_DIRECTION = _descriptor.Descriptor(
  name='Direction',
  full_name='protodrive.Direction',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='op', full_name='protodrive.Direction.op', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='t', full_name='protodrive.Direction.t', index=1,
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
  serialized_start=27,
  serialized_end=61,
)


_POINT = _descriptor.Descriptor(
  name='Point',
  full_name='protodrive.Point',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='protodrive.Point.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='lon', full_name='protodrive.Point.lon', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='lat', full_name='protodrive.Point.lat', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='x', full_name='protodrive.Point.x', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='y', full_name='protodrive.Point.y', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='angle', full_name='protodrive.Point.angle', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='t', full_name='protodrive.Point.t', index=6,
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
  serialized_start=63,
  serialized_end=156,
)


_BRIEFPOINTS = _descriptor.Descriptor(
  name='BriefPoints',
  full_name='protodrive.BriefPoints',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='x1', full_name='protodrive.BriefPoints.x1', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='y1', full_name='protodrive.BriefPoints.y1', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='x2', full_name='protodrive.BriefPoints.x2', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='y2', full_name='protodrive.BriefPoints.y2', index=3,
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
  serialized_start=158,
  serialized_end=219,
)


_STOPINFO = _descriptor.Descriptor(
  name='StopInfo',
  full_name='protodrive.StopInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='isStop', full_name='protodrive.StopInfo.isStop', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='t', full_name='protodrive.StopInfo.t', index=1,
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
  serialized_start=221,
  serialized_end=258,
)


_CHANGEINFO = _descriptor.Descriptor(
  name='ChangeInfo',
  full_name='protodrive.ChangeInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='image', full_name='protodrive.ChangeInfo.image', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='t', full_name='protodrive.ChangeInfo.t', index=1,
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
  serialized_start=260,
  serialized_end=298,
)


_SEINFO = _descriptor.Descriptor(
  name='SEInfo',
  full_name='protodrive.SEInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='state', full_name='protodrive.SEInfo.state', index=0,
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
  serialized_start=300,
  serialized_end=323,
)


_ENDINFO = _descriptor.Descriptor(
  name='EndInfo',
  full_name='protodrive.EndInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='state', full_name='protodrive.EndInfo.state', index=0,
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
  serialized_start=325,
  serialized_end=349,
)

DESCRIPTOR.message_types_by_name['Direction'] = _DIRECTION
DESCRIPTOR.message_types_by_name['Point'] = _POINT
DESCRIPTOR.message_types_by_name['BriefPoints'] = _BRIEFPOINTS
DESCRIPTOR.message_types_by_name['StopInfo'] = _STOPINFO
DESCRIPTOR.message_types_by_name['ChangeInfo'] = _CHANGEINFO
DESCRIPTOR.message_types_by_name['SEInfo'] = _SEINFO
DESCRIPTOR.message_types_by_name['EndInfo'] = _ENDINFO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Direction = _reflection.GeneratedProtocolMessageType('Direction', (_message.Message,), {
  'DESCRIPTOR' : _DIRECTION,
  '__module__' : 'Drive_pb2'
  # @@protoc_insertion_point(class_scope:protodrive.Direction)
  })
_sym_db.RegisterMessage(Direction)

Point = _reflection.GeneratedProtocolMessageType('Point', (_message.Message,), {
  'DESCRIPTOR' : _POINT,
  '__module__' : 'Drive_pb2'
  # @@protoc_insertion_point(class_scope:protodrive.Point)
  })
_sym_db.RegisterMessage(Point)

BriefPoints = _reflection.GeneratedProtocolMessageType('BriefPoints', (_message.Message,), {
  'DESCRIPTOR' : _BRIEFPOINTS,
  '__module__' : 'Drive_pb2'
  # @@protoc_insertion_point(class_scope:protodrive.BriefPoints)
  })
_sym_db.RegisterMessage(BriefPoints)

StopInfo = _reflection.GeneratedProtocolMessageType('StopInfo', (_message.Message,), {
  'DESCRIPTOR' : _STOPINFO,
  '__module__' : 'Drive_pb2'
  # @@protoc_insertion_point(class_scope:protodrive.StopInfo)
  })
_sym_db.RegisterMessage(StopInfo)

ChangeInfo = _reflection.GeneratedProtocolMessageType('ChangeInfo', (_message.Message,), {
  'DESCRIPTOR' : _CHANGEINFO,
  '__module__' : 'Drive_pb2'
  # @@protoc_insertion_point(class_scope:protodrive.ChangeInfo)
  })
_sym_db.RegisterMessage(ChangeInfo)

SEInfo = _reflection.GeneratedProtocolMessageType('SEInfo', (_message.Message,), {
  'DESCRIPTOR' : _SEINFO,
  '__module__' : 'Drive_pb2'
  # @@protoc_insertion_point(class_scope:protodrive.SEInfo)
  })
_sym_db.RegisterMessage(SEInfo)

EndInfo = _reflection.GeneratedProtocolMessageType('EndInfo', (_message.Message,), {
  'DESCRIPTOR' : _ENDINFO,
  '__module__' : 'Drive_pb2'
  # @@protoc_insertion_point(class_scope:protodrive.EndInfo)
  })
_sym_db.RegisterMessage(EndInfo)



_DRIVE = _descriptor.ServiceDescriptor(
  name='Drive',
  full_name='protodrive.Drive',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=352,
  serialized_end=665,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetPosition',
    full_name='protodrive.Drive.GetPosition',
    index=0,
    containing_service=None,
    input_type=_DIRECTION,
    output_type=_POINT,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='DriverStop',
    full_name='protodrive.Drive.DriverStop',
    index=1,
    containing_service=None,
    input_type=_DIRECTION,
    output_type=_STOPINFO,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ChangeImage',
    full_name='protodrive.Drive.ChangeImage',
    index=2,
    containing_service=None,
    input_type=_DIRECTION,
    output_type=_CHANGEINFO,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SendSEPosition',
    full_name='protodrive.Drive.SendSEPosition',
    index=3,
    containing_service=None,
    input_type=_BRIEFPOINTS,
    output_type=_SEINFO,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SendEnd',
    full_name='protodrive.Drive.SendEnd',
    index=4,
    containing_service=None,
    input_type=_ENDINFO,
    output_type=_ENDINFO,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_DRIVE)

DESCRIPTOR.services_by_name['Drive'] = _DRIVE

# @@protoc_insertion_point(module_scope)
