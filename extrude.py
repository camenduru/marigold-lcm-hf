import math
import os

import numpy as np
import pygltflib
import trimesh
from PIL import Image, ImageFilter


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def glb_add_lights(path_input, path_output):
    """
    Adds directional lights in the horizontal plane to the glb file.
    :param path_input: path to input glb
    :param path_output: path to output glb
    :return: None
    """
    glb = pygltflib.GLTF2().load(path_input)

    N = 3  # default max num lights in Babylon.js is 4
    angle_step = 2 * math.pi / N
    elevation_angle = math.radians(75)

    light_colors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    lights_extension = {
        "lights": [
            {"type": "directional", "color": light_colors[i], "intensity": 2.0}
            for i in range(N)
        ]
    }

    if "KHR_lights_punctual" not in glb.extensionsUsed:
        glb.extensionsUsed.append("KHR_lights_punctual")
    glb.extensions["KHR_lights_punctual"] = lights_extension

    light_nodes = []
    for i in range(N):
        angle = i * angle_step

        pos_rot = [0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)]
        elev_rot = [
            math.sin(elevation_angle / 2),
            0.0,
            0.0,
            math.cos(elevation_angle / 2),
        ]
        rotation = quaternion_multiply(pos_rot, elev_rot)

        node = {
            "rotation": rotation,
            "extensions": {"KHR_lights_punctual": {"light": i}},
        }
        light_nodes.append(node)

    light_node_indices = list(range(len(glb.nodes), len(glb.nodes) + N))
    glb.nodes.extend(light_nodes)

    root_node_index = glb.scenes[glb.scene].nodes[0]
    root_node = glb.nodes[root_node_index]
    if hasattr(root_node, "children"):
        root_node.children.extend(light_node_indices)
    else:
        root_node.children = light_node_indices

    glb.save(path_output)


def extrude_depth_3d(
    path_rgb,
    path_depth,
    output_model_scale=100,
    filter_size=3,
    coef_near=0.0,
    coef_far=1.0,
    emboss=0.3,
    f_thic=0.05,
    f_near=-0.15,
    f_back=0.01,
    vertex_colors=True,
    scene_lights=True,
    prepare_for_3d_printing=False,
):
    f_far_inner = -emboss
    f_far_outer = f_far_inner - f_back

    f_near = max(f_near, f_far_inner)

    depth_image = Image.open(path_depth)
    assert depth_image.mode == "I", depth_image.mode
    depth_image = depth_image.filter(ImageFilter.MedianFilter(size=filter_size))

    w, h = depth_image.size
    d_max = max(w, h)
    depth_image = np.array(depth_image).astype(np.double)
    z_min, z_max = np.min(depth_image), np.max(depth_image)
    depth_image = (depth_image.astype(np.double) - z_min) / (z_max - z_min)
    depth_image[depth_image < coef_near] = coef_near
    depth_image[depth_image > coef_far] = coef_far
    depth_image = emboss * (depth_image - coef_near) / (coef_far - coef_near)
    rgb_image = np.array(
        Image.open(path_rgb).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    )

    w_norm = w / float(d_max - 1)
    h_norm = h / float(d_max - 1)
    w_half = w_norm / 2
    h_half = h_norm / 2

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x / float(d_max - 1) - w_half  # [-w_half, w_half]
    y = -y / float(d_max - 1) + h_half  # [-h_half, h_half]
    z = -depth_image  # -depth_emboss (far) - 0 (near)
    vertices_2d = np.stack((x, y, z), axis=-1)
    vertices = vertices_2d.reshape(-1, 3)
    colors = rgb_image[:, :, :3].reshape(-1, 3) / 255.0

    faces = []
    for y in range(h - 1):
        for x in range(w - 1):
            idx = y * w + x
            faces.append([idx, idx + w, idx + 1])
            faces.append([idx + 1, idx + w, idx + 1 + w])

    # OUTER frame

    nv = len(vertices)
    vertices = np.append(
        vertices,
        [
            [-w_half - f_thic, -h_half - f_thic, f_near],  # 00
            [-w_half - f_thic, -h_half - f_thic, f_far_outer],  # 01
            [w_half + f_thic, -h_half - f_thic, f_near],  # 02
            [w_half + f_thic, -h_half - f_thic, f_far_outer],  # 03
            [w_half + f_thic, h_half + f_thic, f_near],  # 04
            [w_half + f_thic, h_half + f_thic, f_far_outer],  # 05
            [-w_half - f_thic, h_half + f_thic, f_near],  # 06
            [-w_half - f_thic, h_half + f_thic, f_far_outer],  # 07
        ],
        axis=0,
    )
    faces.extend(
        [
            [nv + 0, nv + 1, nv + 2],
            [nv + 2, nv + 1, nv + 3],
            [nv + 2, nv + 3, nv + 4],
            [nv + 4, nv + 3, nv + 5],
            [nv + 4, nv + 5, nv + 6],
            [nv + 6, nv + 5, nv + 7],
            [nv + 6, nv + 7, nv + 0],
            [nv + 0, nv + 7, nv + 1],
        ]
    )
    colors = np.append(colors, [[0.5, 0.5, 0.5]] * 8, axis=0)

    # INNER frame

    nv = len(vertices)
    vertices_left_data = vertices_2d[:, 0]  # H x 3
    vertices_left_frame = vertices_2d[:, 0].copy()  # H x 3
    vertices_left_frame[:, 2] = f_near
    vertices = np.append(vertices, vertices_left_data, axis=0)
    vertices = np.append(vertices, vertices_left_frame, axis=0)
    colors = np.append(colors, [[0.5, 0.5, 0.5]] * (2 * h), axis=0)
    for i in range(h - 1):
        nvi_d = nv + i
        nvi_f = nvi_d + h
        faces.append([nvi_d, nvi_f, nvi_d + 1])
        faces.append([nvi_d + 1, nvi_f, nvi_f + 1])

    nv = len(vertices)
    vertices_right_data = vertices_2d[:, -1]  # H x 3
    vertices_right_frame = vertices_2d[:, -1].copy()  # H x 3
    vertices_right_frame[:, 2] = f_near
    vertices = np.append(vertices, vertices_right_data, axis=0)
    vertices = np.append(vertices, vertices_right_frame, axis=0)
    colors = np.append(colors, [[0.5, 0.5, 0.5]] * (2 * h), axis=0)
    for i in range(h - 1):
        nvi_d = nv + i
        nvi_f = nvi_d + h
        faces.append([nvi_d, nvi_d + 1, nvi_f])
        faces.append([nvi_d + 1, nvi_f + 1, nvi_f])

    nv = len(vertices)
    vertices_top_data = vertices_2d[0, :]  # H x 3
    vertices_top_frame = vertices_2d[0, :].copy()  # H x 3
    vertices_top_frame[:, 2] = f_near
    vertices = np.append(vertices, vertices_top_data, axis=0)
    vertices = np.append(vertices, vertices_top_frame, axis=0)
    colors = np.append(colors, [[0.5, 0.5, 0.5]] * (2 * w), axis=0)
    for i in range(w - 1):
        nvi_d = nv + i
        nvi_f = nvi_d + w
        faces.append([nvi_d, nvi_d + 1, nvi_f])
        faces.append([nvi_d + 1, nvi_f + 1, nvi_f])

    nv = len(vertices)
    vertices_bottom_data = vertices_2d[-1, :]  # H x 3
    vertices_bottom_frame = vertices_2d[-1, :].copy()  # H x 3
    vertices_bottom_frame[:, 2] = f_near
    vertices = np.append(vertices, vertices_bottom_data, axis=0)
    vertices = np.append(vertices, vertices_bottom_frame, axis=0)
    colors = np.append(colors, [[0.5, 0.5, 0.5]] * (2 * w), axis=0)
    for i in range(w - 1):
        nvi_d = nv + i
        nvi_f = nvi_d + w
        faces.append([nvi_d, nvi_f, nvi_d + 1])
        faces.append([nvi_d + 1, nvi_f, nvi_f + 1])

    # FRONT frame

    nv = len(vertices)
    vertices = np.append(
        vertices,
        [
            [-w_half - f_thic, -h_half - f_thic, f_near],
            [-w_half - f_thic, h_half + f_thic, f_near],
        ],
        axis=0,
    )
    vertices = np.append(vertices, vertices_left_frame, axis=0)
    colors = np.append(colors, [[0.5, 0.5, 0.5]] * (2 + h), axis=0)
    for i in range(h - 1):
        faces.append([nv, nv + 2 + i + 1, nv + 2 + i])
    faces.append([nv, nv + 2, nv + 1])

    nv = len(vertices)
    vertices = np.append(
        vertices,
        [
            [w_half + f_thic, h_half + f_thic, f_near],
            [w_half + f_thic, -h_half - f_thic, f_near],
        ],
        axis=0,
    )
    vertices = np.append(vertices, vertices_right_frame, axis=0)
    colors = np.append(colors, [[0.5, 0.5, 0.5]] * (2 + h), axis=0)
    for i in range(h - 1):
        faces.append([nv, nv + 2 + i, nv + 2 + i + 1])
    faces.append([nv, nv + h + 1, nv + 1])

    nv = len(vertices)
    vertices = np.append(
        vertices,
        [
            [w_half + f_thic, h_half + f_thic, f_near],
            [-w_half - f_thic, h_half + f_thic, f_near],
        ],
        axis=0,
    )
    vertices = np.append(vertices, vertices_top_frame, axis=0)
    colors = np.append(colors, [[0.5, 0.5, 0.5]] * (2 + w), axis=0)
    for i in range(w - 1):
        faces.append([nv, nv + 2 + i, nv + 2 + i + 1])
    faces.append([nv, nv + 1, nv + 2])

    nv = len(vertices)
    vertices = np.append(
        vertices,
        [
            [-w_half - f_thic, -h_half - f_thic, f_near],
            [w_half + f_thic, -h_half - f_thic, f_near],
        ],
        axis=0,
    )
    vertices = np.append(vertices, vertices_bottom_frame, axis=0)
    colors = np.append(colors, [[0.5, 0.5, 0.5]] * (2 + w), axis=0)
    for i in range(w - 1):
        faces.append([nv, nv + 2 + i + 1, nv + 2 + i])
    faces.append([nv, nv + 1, nv + w + 1])

    # BACK frame

    nv = len(vertices)
    vertices = np.append(
        vertices,
        [
            [-w_half - f_thic, -h_half - f_thic, f_far_outer],  # 00
            [w_half + f_thic, -h_half - f_thic, f_far_outer],  # 01
            [w_half + f_thic, h_half + f_thic, f_far_outer],  # 02
            [-w_half - f_thic, h_half + f_thic, f_far_outer],  # 03
        ],
        axis=0,
    )
    faces.extend(
        [
            [nv + 0, nv + 2, nv + 1],
            [nv + 2, nv + 0, nv + 3],
        ]
    )
    colors = np.append(colors, [[0.5, 0.5, 0.5]] * 4, axis=0)

    trimesh_kwargs = {}
    if vertex_colors:
        trimesh_kwargs["vertex_colors"] = colors
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, **trimesh_kwargs)

    mesh.merge_vertices()

    current_max_dimension = max(mesh.extents)
    scaling_factor = output_model_scale / current_max_dimension
    mesh.apply_scale(scaling_factor)

    if prepare_for_3d_printing:
        rotation_mat = trimesh.transformations.rotation_matrix(np.radians(90), [-1, 0, 0])
        mesh.apply_transform(rotation_mat)

    path_out_base = os.path.splitext(path_depth)[0].replace("_16bit", "")
    path_out_glb = path_out_base + ".glb"
    path_out_stl = path_out_base + ".stl"

    mesh.export(path_out_glb, file_type="glb")
    if scene_lights:
        glb_add_lights(path_out_glb, path_out_glb)

    mesh.export(path_out_stl, file_type="stl")

    return path_out_glb, path_out_stl
