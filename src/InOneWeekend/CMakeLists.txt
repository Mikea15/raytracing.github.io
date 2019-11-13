cmake_minimum_required (VERSION 3.8)

set(MODULE_NAME RayTracerOneWeekend)

# Add source to this project's executable.
add_executable (
	${MODULE_NAME}
	
    camera.h
    hittable_list.h
    hittable.h
	main.cc
    material.h
    random.h
    ray.h
    sphere.h
    vec3.h
)


	