# curl examples (direct Slicer WebServer)

## Render a slice

```bash
curl "http://localhost:2016/slicer/slice?view=axial&orientation=axial&scrollTo=0.5&size=512" -o slice.png
```

## Execute Python (dangerous)

```bash
curl -X POST "http://localhost:2016/slicer/exec" --data "import SampleData; v=SampleData.SampleDataLogic().downloadMRHead(); __execResult={'id': v.GetID()}"
```

## List volumes

```bash
curl "http://localhost:2016/slicer/volumes"
```
