from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration

def setup(context, *args, **kwargs):
    name = LaunchConfiguration("name").perform(context)
    print(f"Launching process for {name}")
    return [ExecuteProcess(cmd=["echo", f"Hello {name}"], output="screen")]

def generate_launch_description():
    ld = LaunchDescription()

    
    # Declare runtime argument
    name_arg = DeclareLaunchArgument(
        "name",
        default_value="World",
        description="Who to greet"
    )
    ld.add_action(name_arg)


    # Add opaque function (dynamic part)
    ld.add_action(OpaqueFunction(function=setup))


    return ld

    # return LaunchDescription([
    #     DeclareLaunchArgument("name", default_value="World"),
    #     OpaqueFunction(function=setup),
    # ])
