## Notes

Some cases that the merge corner will fail.

## The special corner

```
└┐
```

## Mar 29

- [ ] debug why it lose the rect shape
- [ ] debug the vel vector
- [ ] EPE loss
- [ ] add video combiner
- [ ] organize codes, add logging system
- [ ] optimize the raycasting algorithm
- [ ] test on more cases
- [ ] comparison with previous code

## Raycasting-based algorithm

The smallest set to recovery a mask

1. edge segments.
2. edge direction vector.
3. polygon ids.
4. metadata: image width / height

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

```python
        vertices = torch.tensor([[574., 478.],
[574., 802.],
[706., 802.],
[706., 738.],
[638., 738.],
[638., 478.]]).unsqueeze(0)
```

Use the right sequence, it can be good.

______________________________________________________________________

New solution

When construct the polygon, get to the edge.
Record the relevant information.

______________________________________________________________________

```text
BEGIN     /* GL1TOGULP CALLED ON FRI MAY 17 11:33:25 2013 */
EQUIV  1  1000  MICRON  +X,+Y
CNAME Temp_Top
LEVEL M1

CELL Temp_Top PRIME
   RECT N M1  80  80  252  126
   RECT N M1  80  250  256  126
ENDMSG
```

polygon

```python
[[[80, 80], [80, 206], [332, 206], [332, 80]], [[80, 250], [80, 376], [336, 376], [336, 250]]]
```

reshape

```
[[[128, 108], [128, 234], [380, 234], [380, 108]], [[128, 278], [128, 404], [384, 404], [384, 278]]]
```

______________________________________________________________________

test polygon

```python
vertices = torch.tensor(
    [
        [200, 200],
        [900, 200],
        [900, 900],
        [200, 900],
    ],
    dtype=torch.float32,
    device=DEVICE,
)
```

______________________________________________________________________

这是一个tensor表示的两条边：

```python
[[[ 680.,  680.],          [1046., 1070.]],          [[ 680.,  680.],          [1070., 1110.]]]
```

______________________________________________________________________

edge 的向量表示 \[N, 2, 2\] ,

N个edge，2：起点和终点，2：2-D（x,y)

```python
[[x1,x2],[y1,y2]]
```

first edge:

```python
tensor([[ 680.,  680.],
[1070., 1110.]], device='cuda:0', grad_fn=<SelectBackward0>)
dir: tensor([0., 1.], device='cuda:0')
vel: tensor([1., -0.], device='cuda:0')
```

______________________________________________________________________

Better edge view

```python
    print("edges")
    edge_view = edges.clone().detach().transpose(1, 2)
    print(edge_view)
```
