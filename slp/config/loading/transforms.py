from slp.transforms.pipelines import normalize_and_flatten_pipeline


def load_pose_transforms_pipeline(pipeline_name: str):
    if pipeline_name == 'none':
        return lambda x: x
    elif pipeline_name == 'normalize_and_flatten':
        return normalize_and_flatten_pipeline()
    else:
        raise ValueError(f'Unknown pipeline name {pipeline_name}')
