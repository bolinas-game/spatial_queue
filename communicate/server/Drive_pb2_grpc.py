# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import communicate.server.Drive_pb2 as Drive__pb2


class DriveStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetPosition = channel.unary_stream(
                '/protodrive.Drive/GetPosition',
                request_serializer=Drive__pb2.Direction.SerializeToString,
                response_deserializer=Drive__pb2.Point.FromString,
                )
        self.DriverStop = channel.unary_unary(
                '/protodrive.Drive/DriverStop',
                request_serializer=Drive__pb2.Direction.SerializeToString,
                response_deserializer=Drive__pb2.StopInfo.FromString,
                )
        self.ChangeImage = channel.unary_unary(
                '/protodrive.Drive/ChangeImage',
                request_serializer=Drive__pb2.Direction.SerializeToString,
                response_deserializer=Drive__pb2.ChangeInfo.FromString,
                )
        self.SendSEPosition = channel.unary_unary(
                '/protodrive.Drive/SendSEPosition',
                request_serializer=Drive__pb2.BriefPoints.SerializeToString,
                response_deserializer=Drive__pb2.SEInfo.FromString,
                )
        self.SendEnd = channel.unary_unary(
                '/protodrive.Drive/SendEnd',
                request_serializer=Drive__pb2.EndInfo.SerializeToString,
                response_deserializer=Drive__pb2.EndInfo.FromString,
                )


class DriveServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetPosition(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DriverStop(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ChangeImage(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendSEPosition(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendEnd(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DriveServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetPosition': grpc.unary_stream_rpc_method_handler(
                    servicer.GetPosition,
                    request_deserializer=Drive__pb2.Direction.FromString,
                    response_serializer=Drive__pb2.Point.SerializeToString,
            ),
            'DriverStop': grpc.unary_unary_rpc_method_handler(
                    servicer.DriverStop,
                    request_deserializer=Drive__pb2.Direction.FromString,
                    response_serializer=Drive__pb2.StopInfo.SerializeToString,
            ),
            'ChangeImage': grpc.unary_unary_rpc_method_handler(
                    servicer.ChangeImage,
                    request_deserializer=Drive__pb2.Direction.FromString,
                    response_serializer=Drive__pb2.ChangeInfo.SerializeToString,
            ),
            'SendSEPosition': grpc.unary_unary_rpc_method_handler(
                    servicer.SendSEPosition,
                    request_deserializer=Drive__pb2.BriefPoints.FromString,
                    response_serializer=Drive__pb2.SEInfo.SerializeToString,
            ),
            'SendEnd': grpc.unary_unary_rpc_method_handler(
                    servicer.SendEnd,
                    request_deserializer=Drive__pb2.EndInfo.FromString,
                    response_serializer=Drive__pb2.EndInfo.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'protodrive.Drive', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Drive(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetPosition(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/protodrive.Drive/GetPosition',
            Drive__pb2.Direction.SerializeToString,
            Drive__pb2.Point.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DriverStop(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/protodrive.Drive/DriverStop',
            Drive__pb2.Direction.SerializeToString,
            Drive__pb2.StopInfo.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ChangeImage(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/protodrive.Drive/ChangeImage',
            Drive__pb2.Direction.SerializeToString,
            Drive__pb2.ChangeInfo.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SendSEPosition(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/protodrive.Drive/SendSEPosition',
            Drive__pb2.BriefPoints.SerializeToString,
            Drive__pb2.SEInfo.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SendEnd(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/protodrive.Drive/SendEnd',
            Drive__pb2.EndInfo.SerializeToString,
            Drive__pb2.EndInfo.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
