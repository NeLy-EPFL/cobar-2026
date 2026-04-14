import uuid


class GrassMixin:
    @staticmethod
    def _get_square_pyramid_vertices(base_size, height):
        half_base = base_size / 2
        vertices = [
            [-half_base, -half_base, 0],
            [half_base, -half_base, 0],
            [half_base, half_base, 0],
            [-half_base, half_base, 0],
            [0, 0, height],
        ]
        return sum(vertices, [])

    def add_grass_blade(
        self,
        pos,
        base_size=1,
        height=5,
        rgba=(0, 1, 0, 1),
        stiffness=50,
        damping=1,
        mass=0.003,
        **kwargs,
    ):
        name = uuid.uuid4().hex
        vertices = self._get_square_pyramid_vertices(base_size=base_size, height=height)
        mesh = self.mjcf_root.asset.add(
            "mesh",
            name=name,
            vertex=vertices,
        )
        body = self.mjcf_root.worldbody.add(
            "body",
            name=name,
            pos=pos,
        )
        body.add(
            "joint",
            name=f"{name}_x",
            type="hinge",
            axis=(1, 0, 0),
            stiffness=stiffness,
            damping=damping,
        )
        body.add(
            "joint",
            name=f"{name}_y",
            type="hinge",
            axis=(0, 1, 0),
            stiffness=stiffness,
            damping=damping,
        )
        geom = body.add(
            "geom",
            pos=(0, 0, 0),
            name=name,
            type="mesh",
            mesh=mesh,
            contype=0,
            conaffinity=0,
            rgba=rgba,
            mass=mass,
        )
        self.ground_contact_geoms.append(geom)
