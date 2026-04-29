# Tarot 图片目录

把每张牌的图片放在这个目录即可。

## 推荐命名（无需改 JSON）
- 文件名使用 `card_id`，例如：`major_20_judgement.png`
- 支持后缀：`.png` `.jpg` `.jpeg` `.webp` `.gif` `.bmp`

## 可选：在 `tarot_cards.json` 里手动绑定图片
- 你可以在某张牌上加字段：`"image": "你的文件名.png"`
- 也支持子目录：`"image": "major/judgement.png"`

## 逆位图片
- 抽到逆位时会自动将原图旋转 180° 并发送。
- 旋转后的缓存图会放在本目录的 `_generated/` 子目录。
