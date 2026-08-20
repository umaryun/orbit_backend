-- 002_add_reminders_and_preferred_name.sql
-- Add preferred_name to users and create reminders table

-- 1. Add preferred_name to users table if not exists
ALTER TABLE users ADD COLUMN IF NOT EXISTS preferred_name VARCHAR(100);

-- 2. Create reminder_status_enum if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'reminder_status_enum') THEN
        CREATE TYPE reminder_status_enum AS ENUM ('pending', 'delivered', 'cancelled', 'failed');
    END IF;
END$$;

-- 3. Create reminders table
CREATE TABLE IF NOT EXISTS reminders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    message TEXT NOT NULL,
    remind_at TIMESTAMP WITH TIME ZONE NOT NULL,
    status reminder_status_enum NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Create indices
CREATE INDEX IF NOT EXISTS ix_reminders_status_remind_at ON reminders(status, remind_at);
CREATE INDEX IF NOT EXISTS ix_reminders_user_id ON reminders(user_id);
