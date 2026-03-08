# remove-background

Standalone ballot-paper cropper for first-step background removal.

## Run in isolation

```bash
python remove-background/cli.py --input uploads --out_dir runtime-test/remove-background/crops --debug_dir runtime-test/remove-background/debug
```

Debug output per image includes:
- `original_input.png`
- `detected_region.png`
- `warped_before_mask.png`
- `paper_mask.png`
- `final_cropped_output.png`
- `crop_meta.json`
