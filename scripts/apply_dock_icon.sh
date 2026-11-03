#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ICNS_FILE="$PROJECT_ROOT/resources/TimeCapsule.icns"
APP_LOCAL="$PROJECT_ROOT/Time Capsule.app"
APP_SYSTEM="/Applications/Time Capsule.app"

echo "Applying custom Time Capsule icon to applications..."

apply_icon_to_app() {
  local target_app="$1"
  if [ ! -d "$target_app" ]; then
    echo "Creating target app at $target_app..."
    cp -R "$APP_LOCAL" "$target_app"
  fi

  # 1. Remove Asset Catalog so macOS falls back directly to applet.icns
  rm -f "$target_app/Contents/Resources/Assets.car"

  # 2. Copy the high-res ICNS file
  cp -f "$ICNS_FILE" "$target_app/Contents/Resources/applet.icns"

  # 3. Clean Info.plist (remove CFBundleIconName, ensure CFBundleIconFile is applet.icns)
  local plist="$target_app/Contents/Info.plist"
  /usr/libexec/PlistBuddy -c "Delete :CFBundleIconName" "$plist" 2>/dev/null || true
  /usr/libexec/PlistBuddy -c "Set :CFBundleIconFile applet.icns" "$plist" 2>/dev/null || \
  /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string applet.icns" "$plist" 2>/dev/null || true

  # 4. Use JXA Cocoa NSWorkspace to stamp the custom icon onto the bundle
  osascript -l JavaScript -e "
    ObjC.import('AppKit');
    var ws = $.NSWorkspace.sharedWorkspace;
    var img = $.NSImage.alloc.initWithContentsOfFile('$ICNS_FILE');
    ws.setIconForFileOptions(img, '$target_app', 0);
  " 2>/dev/null || true

  # 5. Set Finder custom icon bit using SetFile
  if [ -x /usr/bin/SetFile ]; then
    /usr/bin/SetFile -a C "$target_app" 2>/dev/null || true
  fi

  touch "$target_app"
  echo "✔ Successfully updated icon for: $target_app"
}

apply_icon_to_app "$APP_LOCAL"
apply_icon_to_app "$APP_SYSTEM"
apply_icon_to_app "$PROJECT_ROOT/Start Time Capsule.app"
apply_icon_to_app "$PROJECT_ROOT/Stop Time Capsule.app"
[ -d "$PROJECT_ROOT/../Start Time Capsule.app" ] && apply_icon_to_app "$PROJECT_ROOT/../Start Time Capsule.app"
[ -d "$PROJECT_ROOT/../Stop Time Capsule.app" ] && apply_icon_to_app "$PROJECT_ROOT/../Stop Time Capsule.app"

# Invalidate LaunchServices cache
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister"
if [ -x "$LSREGISTER" ]; then
  "$LSREGISTER" -f -R "$APP_SYSTEM" 2>/dev/null || true
  "$LSREGISTER" -f -R "$APP_LOCAL" 2>/dev/null || true
fi

# Ensure Dock persistent-apps entry is present and refreshed
echo "Refreshing Dock persistent application tray..."
if ! defaults read com.apple.dock persistent-apps | grep -q "Time Capsule"; then
  defaults write com.apple.dock persistent-apps -array-add "<dict><key>tile-data</key><dict><key>file-data</key><dict><key>_CFURLString</key><string>$APP_SYSTEM/</string><key>_CFURLStringType</key><integer>0</integer></dict><key>file-label</key><string>Time Capsule</string></dict><key>tile-type</key><string>file-tile</string></dict>"
fi

# Restart Dock and Finder to flush icon cache
killall Dock 2>/dev/null || true

echo "✔ Time Capsule icon applied and Dock refreshed successfully."
