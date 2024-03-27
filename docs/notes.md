## Tree construction

V-edges: easy to get linked list

H-edges: easy to get linked list

start
end
next

```python
            segments.append(
                {
                    "segment": torch.stack([start_point, end_point], dim=1),
                    "type": seg_type_label,
                    "id": segment_id,
                    "start": False,
                    "end": True,
                    "next": "To next edge"
                }
            )
```

The segments structure.

```
->->->->|
|       |
|       |
|->->->->
```

______________________________________________________________________

LevelSet

```python
mask = self._binarize(params)
printedNom, printedMax, printedMin = self._lithosim(mask)
```

______________________________________________________________________

Todo

seg_params to polygon and binary images

```

    all_polygons_by_all_id = []
    for polygon in all_polygons_by_start_id:
        polygon_all_ids = []
        for seg_id in polygon:
            edge_list = get_sub_by_start(all_traverse_list, seg_id)
            polygon_all_ids.extend(edge_list)

```

```

    all_polygons_by_start_id = [[element for tuple_pair in sublist for element in tuple_pair] for sublist in all_polygons_by_start_pair]
    print(all_polygons_by_start_id)



```
